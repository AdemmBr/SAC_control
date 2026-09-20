"""Tableau de performance de Zhang, Zohren et Roberts (2019), section 4.4.

Les neuf indicateurs du papier "Deep Reinforcement Learning for Trading"
(arXiv:1911.10107), calcules sur une serie de rendements quotidiens nets.
Les noms de colonnes reprennent ceux du papier pour que le tableau produit ici
se lise en regard de ses tables 2 et 3 sans traduction intermediaire.
"""

import numpy as np
import pandas as pd


class PaperMetrics:
    """Calcule les neuf indicateurs du papier pour chaque colonne de rendements."""

    # Ordre des colonnes du papier. Il fixe aussi l'ordre d'affichage.
    COLUMNS = [
        "E(R)",
        "Std(R)",
        "DD",
        "Sharpe",
        "Sortino",
        "MDD",
        "Calmar",
        "% +ve Returns",
        "Ave. P / Ave. L",
    ]

    # Les deux lectures possibles de la deviation a la baisse. Voir
    # _downside_deviation : elles ne mesurent pas la meme chose, et un Sortino
    # publie n'a de sens qu'accompagne de celle qui l'a produit.
    DOWNSIDE_DEFINITIONS = ("semi_deviation", "negative_std")

    def __init__(
        self, trading_days: int = 252, downside: str = "semi_deviation"
    ):
        """Memorise l'annualisation et la lecture retenue pour DD."""
        if downside not in self.DOWNSIDE_DEFINITIONS:
            raise ValueError(
                f"downside doit valoir l'un de {self.DOWNSIDE_DEFINITIONS}, "
                f"recu {downside!r}"
            )
        self.trading_days = trading_days
        self.downside = downside

    def table(self, returns: pd.DataFrame | pd.Series):
        """Retourne une ligne par strategie et une colonne par indicateur."""
        frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
        root = np.sqrt(self.trading_days)

        # E(R) est annualise arithmetiquement, moyenne x 252, comme dans le papier.
        # Ce n'est pas annual_return de _stats, qui compose : les deux ne coincident
        # que si les rendements sont infinitesimaux, et l'ecart grandit avec la vol.
        expected = frame.mean() * self.trading_days
        volatility = frame.std() * root
        downside = self._downside_deviation(frame) * root
        drawdown = self._max_drawdown(frame)

        gains = frame.where(frame > 0.0).mean()
        losses = frame.where(frame < 0.0).mean()
        # Denominateur du pourcentage : les seances observees, pas les lignes du
        # tableau. Un bloc avec des trous ne doit pas voir son ratio dilue.
        observed = self._nonzero(frame.notna().sum())
        positive = frame.gt(0.0).sum()

        table = pd.DataFrame(
            {
                "E(R)": expected,
                "Std(R)": volatility,
                "DD": downside,
                "Sharpe": expected / self._nonzero(volatility),
                "Sortino": expected / self._nonzero(downside),
                "MDD": drawdown,
                # Calmar garde le E(R) arithmetique au numerateur. Les tables
                # 3 et 11 de Lim, Zohren et Roberts se reproduisent exactement
                # ainsi, et pas avec un rendement compose : 0.117 / 0.431 = 0.271
                # pour leur Long Only, chiffre publie. empyrical.calmar_ratio,
                # lui, compose son numerateur et donnerait 0.244 sur cette ligne.
                "Calmar": expected / self._nonzero(drawdown),
                "% +ve Returns": 100.0 * positive / observed,
                "Ave. P / Ave. L": gains / self._nonzero(losses.abs()),
            }
        )
        return table[self.COLUMNS]

    def _downside_deviation(self, frame: pd.DataFrame):
        """Deviation a la baisse, selon la lecture choisie a la construction.

        "semi_deviation" est la racine du moment d'ordre deux de min(0, r) sur
        toutes les seances, centree sur zero. Elle mesure l'amplitude des pertes.
        C'est la definition usuelle du Sortino, et celle que calcule reellement
        le papier : son implementation de reference appelle empyrical, dont
        downside_risk est exactement cette formule.

        "negative_std" suit la lettre de sa prose, "annualised standard deviation
        of trade returns that are negative" : l'ecart-type du sous-echantillon
        negatif, centre sur la moyenne de ce sous-echantillon. Cela mesure la
        dispersion des pertes et non leur amplitude, au point que des pertes
        toutes egales donnent zero, et un Sortino non defini, quelle que soit
        leur taille. La phrase decrit donc mal la formule employee.

        Deux elements ont tranche. Le code de reference du groupe importe
        empyrical.downside_risk. Et le rapport DD/Vol publie dans les tables de
        Lim, Zohren et Roberts vaut 0.662 a 0.684 : la semi-deviation rend cette
        fourchette pour toute epaisseur de queue, la lecture litterale ne la
        rend que sous une hypothese precise et donne 0.59 sous une gaussienne.

        L'ecart-type garde le ddof=1 de pandas, comme partout dans le projet : un
        bloc qui ne contient qu'une seule seance perdante donne NaN, et non zero.
        Un echantillon de taille un ne porte aucune estimation de dispersion.
        """
        if self.downside == "semi_deviation":
            return frame.clip(upper=0.0).pow(2).mean().pow(0.5)
        return frame.where(frame < 0.0).std()

    def _max_drawdown(self, frame: pd.DataFrame):
        """Perte maximale depuis un sommet, rendue en magnitude positive.

        Le signe positif est celui du papier et il est ce qui rend Calmar positif
        pour une strategie gagnante. Le max_drawdown de WalkForwardRunner._stats
        garde a l'inverse son signe negatif, convention deja en place dans les
        tableaux du notebook : les deux colonnes sont la meme quantite, opposee.

        La courbe de capital compose, comme partout ailleurs dans le projet. Le
        papier travaille sur des rendements de futures additifs ; l'ecart reste
        du second ordre aux niveaux de volatilite en jeu ici.
        """
        equity = (1.0 + frame).cumprod()
        # Le + 0.0 ramene le zero negatif que produit la negation d'un minimum nul
        # a un zero ordinaire, sinon une serie sans repli s'affiche en "-0.0000".
        return -(equity / equity.cummax() - 1.0).min() + 0.0

    @staticmethod
    def _nonzero(values: pd.Series):
        """Neutralise les denominateurs nuls, qui donneraient un ratio infini.

        Un denominateur nul veut dire que l'echantillon ne contient pas ce qu'on
        mesure : aucun jour perdant, ou aucun repli. NaN dit cela ; l'infini le
        ferait passer pour une performance.
        """
        return values.replace(0.0, np.nan)
