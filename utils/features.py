"""Features de marche observees par SAC.
"""

import numpy as np
import pandas as pd


class AgentFeatures:
    """Calcule et detient les features de marche de l'agent."""

    # Liste explicite et ordonnee des features du state, dans l'ordre des colonnes
    # de la matrice F. Elle seule decide de ce que l'agent observe : un attribut
    # pose sur l'objet n'entre pas dans le state tant qu'il n'est pas nomme ici.
    # L'inventaire precedent parcourait vars(self), donc un attribut public ajoute
    # par megarde devenait une feature constante, standardisee en une colonne de
    # zeros sans qu'aucune erreur ne soit levee.
    FEATURES = [
        "momentum_21",
        "momentum_63",
        "return_z",
        "vol_regime",
        "vol_short_long_ratio",
        "downside_vol_ratio",
        "market_drawdown",
    ]

    def __init__(self, vol_window: int = 21, regime_window: int = 252):
        """Memorise les fenetres de l'agent, distinctes de celles de la baseline."""
        self._vol_window = vol_window
        self._regime_window = regime_window

    def build(self, market_data: pd.DataFrame):
        """Calcule les features depuis price/ret et retourne leur seul tableau."""
        returns = market_data["ret"]
        self._volatility = (
            returns.rolling(self._vol_window).std() * np.sqrt(252)
        ).replace(0.0, np.nan)

        self.momentum_21 = self._momentum(returns, 21)
        self.momentum_63 = self._momentum(returns, 63)

        daily_volatility = self._volatility.shift(1) / np.sqrt(252)
        self.return_z = (returns / daily_volatility).clip(-5.0, 5.0)

        
        self.vol_regime = np.log(
            self._volatility
            / self._volatility.rolling(self._regime_window).median()
        ).clip(-1.5, 1.5)

        short_volatility = returns.rolling(5).std() * np.sqrt(252)
        self.vol_short_long_ratio = (
            short_volatility / self._volatility
        ).clip(0.0, 5.0)

        downside_volatility = (
            returns.clip(upper=0.0).pow(2).rolling(self._vol_window).mean().pow(0.5)
            * np.sqrt(252)
        )
        self.downside_vol_ratio = (
            downside_volatility / self._volatility
        ).clip(0.0, 5.0)

        rolling_high = market_data["price"].rolling(63).max()
        self.market_drawdown = (
            market_data["price"] / rolling_high - 1.0
        ).clip(-1.0, 0.0)

        return self.frame()

    def frame(self):
        """Assemble les features declarees en un seul tableau."""
        missing = [name for name in self.FEATURES if not hasattr(self, name)]
        if missing:
            raise AttributeError(
                f"features declarees mais non calculees par build() : {missing}"
            )
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.FEATURES}
        )

    def _momentum(self, returns: pd.Series, window: int):
        """Momentum normalise par le mouvement attendu sur la fenetre."""
        move = returns.rolling(window).sum()
        expected_move = self._volatility * np.sqrt(window / 252)
        return (move / expected_move).clip(-3.0, 3.0)
