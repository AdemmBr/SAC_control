"""Environnement Gymnasium du controleur d'exposition."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .scaling import StateScaler


class ExposureEnv(gym.Env):
    """Applique les decisions SAC au portefeuille et calcule leur reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        scaler: StateScaler,
        transaction_cost_bps: float = 1.0,
        episode_length: int | None = None,
        a_max: float = 1.0,
        max_exposure: float = 2.0,
        lambda_risk: float = 0.0,
    ):
        """Memorise les donnees et declare l'interface requise par Gymnasium."""
        super().__init__()
        # Le scaler porte les colonnes du state et les statistiques du train.
        self.F = scaler.transform(data).to_numpy(dtype=np.float32)
        prices = data["price"].to_numpy()
        self.R = prices[1:] / prices[:-1] - 1.0
        self.P = data["w_base"].to_numpy()
        self.dates = data.index
        self.cost_rate = transaction_cost_bps * 1e-4
        # None : balayage complet du bloc, utilise en evaluation.
        # Entier : fenetre de cette longueur tiree au hasard, utilisee en entrainement.
        self.episode_length = episode_length
        # A 1.0 SAC ne peut qu'attenuer la baseline. Au-dessus il peut l'amplifier, et
        # le milieu de la boite d'action, qu'une politique non entrainee produit,
        # se deplace de a_max/2 : a 2.0 ce milieu vaut la baseline elle-meme.
        self.a_max = a_max
        # w_max du papier. Doit valoir celui de la baseline : c'est la meme borne
        # d'exposition, reappliquee apres le multiplicateur, pas une seconde borne.
        self.max_exposure = max_exposure
        # Poids de la penalite de risque. A 0.0 la recompense redevient le P&L net
        # pur : c'est l'ablation exacte de ce terme, pas une approximation.
        self.lambda_risk = lambda_risk
        # previous_exposure vit sur l'echelle de w_base : le meme ecart-type de train
        # le ramene au niveau des features standardisees. Aucun recentrage ici, une
        # position nulle doit rester a zero pour le reseau.
        self._exposure_scale = float(scaler.std_["w_base"])

        # Ces deux espaces sont obligatoires pour que SB3 connaisse les dimensions.
        self.action_space = spaces.Box(0.0, a_max, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.F.shape[1] + 1,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Ouvre un bloc d'evaluation, ou tire un episode d'entrainement."""
        super().reset(seed=seed)
        if self.episode_length is None:
            # Evaluation : balayage complet, position heritee du fold precedent.
            self._index = 0
            self._end = len(self.R)
            self._previous_exposure = (options or {}).get("previous_exposure", 0.0)
        else:
            # Entrainement : fenetre et position de depart tirees au hasard, pour que
            # l'agent ne rejoue pas la meme sequence et parte de positions variees.
            # SB3 reinitialise sans options, il n'y a donc rien a restaurer ici.
            self._index = int(
                self.np_random.integers(0, len(self.R) - self.episode_length + 1)
            )
            self._end = self._index + self.episode_length
            self._previous_exposure = self._exposure(
                self.np_random.uniform(0.0, self.a_max)
            )
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        """Applique une action au rendement du jour suivant puis avance d'un jour."""
        multiplier = float(action[0])
        baseline_exposure = self.P[self._index]
        exposure = self._exposure(multiplier)
        asset_return = self.R[self._index]
        turnover = abs(exposure - self._previous_exposure)
        market_pnl = exposure * asset_return
        net_return = market_pnl - self.cost_rate * turnover
        self._previous_exposure = exposure

        # Psi_{t+1} : le carre du P&L de marche estime sans biais sa variance
        # conditionnelle. C'est ce terme qui rend la recompense concave en exposition.
        # Sans lui l'objectif est lineaire, donc son optimum est sur une borne de
        # [0, a_max] et l'agent fait du tout-ou-rien au lieu de moduler.
        # Il ne depend que de l'action et du rendement suivant : aucune variable
        # supplementaire n'entre dans le state, le probleme reste markovien.
        risk_penalty = self.lambda_risk * market_pnl**2

        info = {
            "date": self.dates[self._index],
            "multiplier": multiplier,
            "baseline_exposure": baseline_exposure,
            "exposure": exposure,
            "net_return": net_return,
            "risk_penalty": risk_penalty,
        }

        self._index += 1
        truncated = self._index >= self._end
        # net_return reste le P&L pur : la penalite oriente l'apprentissage, elle
        # n'entre jamais dans les metriques financieres rapportees.
        reward = 100.0 * (net_return - risk_penalty)
        return self._get_observation(), reward, False, truncated, info

    def portfolio_state(self):
        """Transmet la position au fold OOS suivant."""
        return {"previous_exposure": self._previous_exposure}

    def _exposure(self, multiplier: float):
        """Applique w_t = clip(A_t * w_base_t, +/- w_max), le clip externe du papier.

        Il ne mordait jamais tant que a_max valait 1 ; il mord des que l'agent peut
        amplifier, en particulier quand la vol basse a deja pousse w_base a sa borne.
        """
        return float(
            np.clip(
                multiplier * self.P[self._index],
                -self.max_exposure,
                self.max_exposure,
            )
        )

    def _get_observation(self):
        """Retourne les features et la position, ramenees a une echelle commune."""
        return np.append(
            self.F[self._index],
            [self._previous_exposure / self._exposure_scale],
        ).astype(np.float32)
