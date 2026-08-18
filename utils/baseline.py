"""Strategie de reference utilisee avant le controle SAC."""

import numpy as np
import pandas as pd


class TrendVolatilityBaseline:
    """Combine une tendance 63 jours et une cible de volatilite."""

    def __init__(
        self,
        trend_window: int = 63,
        vol_window: int = 21,
        target_vol: float = 0.10,
        max_exposure: float = 2.0,
    ):
        """Definit les hypotheses propres a la baseline."""
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_exposure = max_exposure

    def build(self, data: pd.DataFrame):
        """Ajoute le signal, la volatilite et l'exposition de la baseline."""
        frame = data.copy()
        trend = frame["ret"].rolling(self.trend_window).sum()
        frame["signal"] = np.sign(trend)
        frame["forecast_vol"] = (
            frame["ret"].rolling(self.vol_window).std() * np.sqrt(252)
        )
        volatility = frame["forecast_vol"].replace(0.0, np.nan)
        frame["w_base"] = (
            frame["signal"] * self.target_vol / volatility
        ).clip(-self.max_exposure, self.max_exposure)
        return frame
