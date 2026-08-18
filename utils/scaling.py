"""Standardisation du state a partir des seules statistiques d'entrainement."""

import pandas as pd


class StateScaler:
    """Ajuste moyenne et ecart-type sur le train, puis transforme n'importe quel bloc."""

    def fit(self, train_data: pd.DataFrame, columns: list[str]):
        """Memorise les statistiques du train ; aucune donnee de test n'entre ici."""
        self.columns = columns
        self.mean_ = train_data[columns].mean()
        self.std_ = train_data[columns].std().replace(0.0, 1.0)
        return self

    def transform(self, data: pd.DataFrame):
        """Applique les statistiques du train, y compris a un bloc hors echantillon."""
        return (data[self.columns] - self.mean_) / self.std_
