"""Strategies de reference comparees au controle SAC.

TrendVolatilityBaseline est la baseline historique du projet. Les trois autres
reprennent les baselines de la section 4.2 de Zhang, Zohren et Roberts (2019),
"Deep Reinforcement Learning for Trading" (arXiv:1911.10107) : Long Only,
Sign(R) et le signal MACD de Baz et al.

Toutes partagent la meme mise a l'echelle par la volatilite et la meme borne
d'exposition, definies une fois dans ExposureStrategy. Seul le signal les
distingue, ce qui est la condition pour qu'un ecart de performance s'attribue au
signal et non au budget de risque.
"""

import numpy as np
import pandas as pd


class ExposureStrategy:
    """Ossature commune : un signal directionnel, mis a l'echelle par la vol.

    Une strategie produit un signal dans [-1, 1], que la mise a l'echelle
    transforme en exposition visant target_vol, avant le clip a max_exposure.
    Les sous-classes n'ont que _signal a definir.
    """

    def __init__(
        self,
        vol_window: int = 21,
        target_vol: float = 0.10,
        max_exposure: float = 2.0,
        vol_method: str = "rolling",
    ):
        """Definit le budget de risque, commun a toutes les strategies comparees.

        vol_method "rolling" est la convention du projet, un ecart-type sur
        vol_window seances. "ewm" est celle du papier, une moyenne mobile
        exponentielle de span vol_window, qu'il fixe a 60 seances. Le defaut
        reste celui du projet pour que les quatre strategies partagent le meme
        estimateur : les comparer sous deux estimateurs melangerait l'effet du
        signal et celui du budget de risque.
        """
        if vol_method not in ("rolling", "ewm"):
            raise ValueError(
                f"vol_method doit valoir 'rolling' ou 'ewm', recu {vol_method!r}"
            )
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_exposure = max_exposure
        self.vol_method = vol_method

    def build(self, data: pd.DataFrame):
        """Ajoute le signal, la volatilite et l'exposition de la strategie."""
        frame = data.copy()
        frame["signal"] = self._signal(frame)
        frame["forecast_vol"] = self._forecast_vol(frame)
        frame["w_base"] = self._scale(frame["signal"], frame["forecast_vol"])
        return frame

    def exposure(self, data: pd.DataFrame):
        """Retourne la seule exposition, sous forme de colonne nommable.

        C'est l'interface a utiliser pour poser plusieurs strategies cote a cote
        dans un meme tableau : build() ecrirait trois fois les memes noms de
        colonnes et les strategies s'ecraseraient entre elles.
        """
        return self.build(data)["w_base"]

    def _signal(self, frame: pd.DataFrame):
        """Signal directionnel dans [-1, 1]. Seul point de variation."""
        raise NotImplementedError

    def _forecast_vol(self, frame: pd.DataFrame):
        """Volatilite annualisee servant de denominateur a la mise a l'echelle."""
        returns = frame["ret"]
        if self.vol_method == "rolling":
            deviation = returns.rolling(self.vol_window).std()
        else:
            deviation = returns.ewm(span=self.vol_window).std()
        return deviation * np.sqrt(252)

    def _scale(self, signal: pd.Series, volatility: pd.Series):
        """Vise target_vol puis borne l'exposition, w = clip(s * cible / vol)."""
        return (signal * self.target_vol / volatility.replace(0.0, np.nan)).clip(
            -self.max_exposure, self.max_exposure
        )


class TrendVolatilityBaseline(ExposureStrategy):
    """Combine une tendance 63 jours et une cible de volatilite."""

    def __init__(self, trend_window: int = 63, **kwargs):
        """Definit les hypotheses propres a la baseline."""
        super().__init__(**kwargs)
        self.trend_window = trend_window

    def _signal(self, frame: pd.DataFrame):
        """Signe du rendement cumule sur trend_window seances."""
        return np.sign(frame["ret"].rolling(self.trend_window).sum())


class LongOnly(ExposureStrategy):
    """Long Only du papier : position longue constante, mise a l'echelle.

    Ce n'est pas le Buy_and_hold du runner. Celui-la detient une unite de
    l'actif et ne se rebalance jamais ; celle-ci vise target_vol, donc elle
    monte quand le marche est calme et se replie quand il s'agite. La
    comparaison des deux isole exactement ce que le ciblage de volatilite
    apporte, a signal directionnel identique.
    """

    def _signal(self, frame: pd.DataFrame):
        """Toujours long, sans jamais de vue directionnelle."""
        return pd.Series(1.0, index=frame.index)


class SignReturn(ExposureStrategy):
    """Sign(R) du papier, equation 10 : A_t = sign(r_{t-252:t}).

    Le momentum de serie temporelle classique. Il ne differe de la baseline du
    projet que par sa fenetre, une annee au lieu d'un trimestre.
    """

    def __init__(self, lookback: int = 252, **kwargs):
        """Memorise l'horizon du momentum, une annee civile dans le papier."""
        super().__init__(**kwargs)
        self.lookback = lookback

    def _signal(self, frame: pd.DataFrame):
        """Signe du rendement cumule sur lookback seances."""
        return np.sign(frame["ret"].rolling(self.lookback).sum())


class MACDSignal(ExposureStrategy):
    """Signal MACD de Baz et al., equations 3, 11 et 12 du papier.

    Pour une paire d'echelles (S, L), l'ecart de deux moyennes mobiles
    exponentielles de prix est normalise deux fois : par l'ecart-type des prix
    sur 63 seances, puis par son propre ecart-type sur 252 seances. Les trois
    paires sont moyennees, et la fonction de reponse phi transforme cette
    moyenne en position, en attenuant les signaux extremes plutot qu'en les
    saturant.

    Deux details ne se lisent pas dans le papier seul et viennent de sa source,
    Lim, Zohren et Roberts (2019), arXiv:1904.04912, et de l'implementation de
    reference de ce groupe.

    L'equation 12 ecrit une somme, mais la phrase qui la precede dit "average
    them" ; l'equation 8 de la source porte exactement la meme contradiction, et
    son implementation de reference divise par le nombre de paires. C'est donc
    une moyenne. La lettre de l'equation se refute d'ailleurs seule : la somme de
    trois quantites deja normalisees passe 71 % du temps au-dela de la racine de
    deux, ou phi redescend vers zero, et le signal moyen s'effondre a 0.03 contre
    0.20 pour la moyenne.

    L'agregation porte sur l'estimation de tendance, avant phi, et non sur les
    reponses. C'est l'ordre de la source, ou l'equation 8 combine les Y et
    l'equation 7 applique phi au resultat. L'ordre inverse reste correle a 0.96
    mais s'en ecarte de 0.11 en valeur absolue moyenne, assez pour changer les
    chiffres publies.
    """

    # Paires (court, long) de l'equation 12.
    TIMESCALES = ((8, 24), (16, 48), (32, 96))

    # Normalisation de phi. Le papier la pose a 0.89, ce qui borne la reponse a
    # environ 0.964 en valeur absolue, donc dans la meme plage que sign(R).
    RESPONSE_SCALE = 0.89

    def _signal(self, frame: pd.DataFrame):
        """Reponse phi appliquee a la moyenne des trois estimations de tendance."""
        price = frame["price"]
        trend = sum(
            self._macd(price, short, long_) for short, long_ in self.TIMESCALES
        ) / len(self.TIMESCALES)
        return self._response(trend)

    def _macd(self, price: pd.Series, short: int, long_: int):
        """MACD normalise de l'equation 3, pour une paire d'echelles.

        Les deux ecarts-types glissants laissent des NaN sur les 313 premieres
        seances. L'implementation de reference les remplit par backfill, ce qui
        y fait entrer une valeur future ; ils sont laisses vides ici et le
        notebook retire ces lignes, bien avant le premier bloc d'entrainement.
        """
        spread = self._average(price, short) - self._average(price, long_)
        # Les deux denominateurs sont neutralises a zero, comme la volatilite de
        # _scale : un prix fige sur la fenetre donnerait sinon un signal infini.
        q = spread / price.rolling(63).std().replace(0.0, np.nan)
        return q / q.rolling(252).std().replace(0.0, np.nan)

    @staticmethod
    def _average(price: pd.Series, scale: int):
        """Moyenne mobile exponentielle de prix a l'echelle demandee.

        Le papier ecrit "exponentially weighted moving average of prices with a
        time scale S" sans donner la conversion en demi-vie. Sa source la donne :
        HL = log(0.5) / log(1 - 1/S), la demi-vie dont le facteur de lissage
        vaut 1/S.
        """
        halflife = np.log(0.5) / np.log(1.0 - 1.0 / scale)
        return price.ewm(halflife=halflife).mean()

    @classmethod
    def _response(cls, trend: pd.Series):
        """Fonction de reponse phi de l'equation 11.

        Elle croit jusqu'a la racine de deux puis redescend vers zero : au-dela,
        l'actif est considere comme sur-achete ou sur-vendu et la position est
        reduite, au lieu d'etre poussee a sa borne.
        """
        return trend * np.exp(-(trend**2) / 4.0) / cls.RESPONSE_SCALE
