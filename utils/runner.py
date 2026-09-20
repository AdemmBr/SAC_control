"""Entrainement SAC et evaluation financiere en walk-forward."""

import time

import pandas as pd
import torch
from stable_baselines3 import SAC

from .environment import ExposureEnv
from .metrics import PaperMetrics
from .scaling import StateScaler


class WalkForwardRunner:
    """Entraine SAC sur une fenetre d'annees civiles puis l'evalue sur la suivante."""

    def __init__(
        self,
        sac_config: dict,
        state_columns: list[str],
        first_train_start: str,
        seeds: list[int],
        total_timesteps: int,
        train_years: int = 4,
        test_years: int = 1,
        episode_length: int = 252,
        transaction_cost_bps: float = 1.0,
        rolling_train: bool = True,
        a_max: float = 1.0,
        max_exposure: float = 2.0,
        lambda_risk: float = 0.0,
        baseline_columns: dict[str, str] | None = None,
    ):
        """Memorise les choix du walk-forward et de SAC."""
        self.sac_config = sac_config
        self.state_columns = state_columns
        self.first_train_start = first_train_start
        self.train_years = train_years
        self.test_years = test_years
        self.episode_length = episode_length
        # True : le train garde train_years annees (fenetre glissante).
        # False : le train part toujours de la premiere date (fenetre extensible).
        self.rolling_train = rolling_train
        self.seeds = seeds
        self.total_timesteps = total_timesteps
        self.transaction_cost_bps = transaction_cost_bps
        self.cost_rate = transaction_cost_bps * 1e-4
        # Borne du multiplicateur, et w_max reapplique apres lui. Ce dernier doit
        # valoir le max_exposure de la baseline, sinon les deux bornes divergent.
        self.a_max = a_max
        self.max_exposure = max_exposure
        # Arbitrage rendement / variance, les deux annualises : l'exposition optimale
        # sans cout vaut w* = mu / (2 * lambda_risk * sigma^2). A 0.0, ablation exacte.
        self.lambda_risk = lambda_risk
        # Strategies de reference supplementaires, nom affiche -> colonne
        # d'exposition deja calculee dans data. La baseline historique reste
        # w_base et n'a pas a figurer ici. Chacune est evaluee sur le meme bloc
        # OOS, avec le meme cout de turnover, et entre dans _criterion.
        self.baseline_columns = baseline_columns or {}
        # Source unique du Sharpe, de la volatilite et du drawdown : _stats en
        # reprend quatre colonnes sous les noms deja employes dans le notebook.
        self._paper = PaperMetrics()

        self.folds_ = None
        self.decisions_ = None
        self.daily_returns_ = None
        self.metrics_ = None
        self.fold_metrics_ = None
        self.paper_metrics_ = None
        self.paper_fold_metrics_ = None
        self.criterion_ = None

    def run(self, data: pd.DataFrame):
        """Execute tous les entrainements et construit les resultats OOS."""
        # Le nombre de threads change les resultats : les reductions paralleles de
        # torch ne somment pas dans le meme ordre. Mesure sur un fold, seed 42 :
        # sharpe 0.48 a un thread contre 0.49 a deux. Le fixer ici plutot que par
        # OMP_NUM_THREADS evite que le defaut suive le nombre de coeurs de la machine.
        torch.set_num_threads(1)
        splits = self._splits(data.index)
        fold_rows, paths = [], []
        portfolio_states = {seed: None for seed in self.seeds}

        for fold, (train_start, train_end, test_end) in enumerate(splits):
            # Le train s'arrete a train_end exclu. L'inclure donnait a l'agent la
            # transition dont le rendement se realise le premier jour du test : une
            # journee de lookahead, et une incoherence avec le scaler qui, lui,
            # s'arretait deja a train_end exclu.
            train = data.iloc[train_start:train_end]
            # Le test garde une ligne de plus que son bloc OOS : elle ne sert qu'a
            # calculer le rendement du dernier jour, jamais a decider.
            test = data.iloc[train_end : test_end + 1]
            # Ajuste sur les seules lignes ou l'agent decide pendant l'entrainement.
            scaler = StateScaler().fit(train, self.state_columns)
            print(
                f"Fold {fold} | train {data.index[train_start].date()} -> "
                f"{data.index[train_end - 1].date()} ({train_end - train_start} j) "
                f"| test {data.index[train_end].date()} -> "
                f"{data.index[test_end - 1].date()}",
                flush=True,
            )

            fold_rows.append(
                {
                    "fold": fold,
                    "train_start": data.index[train_start],
                    "train_end": data.index[train_end - 1],
                    "test_start": data.index[train_end],
                    "test_end": data.index[test_end - 1],
                }
            )

            for seed in self.seeds:
                started = time.perf_counter()
                environment = ExposureEnv(
                    train,
                    scaler,
                    self.transaction_cost_bps,
                    episode_length=self.episode_length,
                    a_max=self.a_max,
                    max_exposure=self.max_exposure,
                    lambda_risk=self.lambda_risk,
                )
                model = SAC(
                    "MlpPolicy",
                    environment,
                    seed=seed,
                    verbose=0,
                    **self.sac_config,
                )
                model.learn(total_timesteps=self.total_timesteps)

                environment = ExposureEnv(
                    test,
                    scaler,
                    self.transaction_cost_bps,
                    episode_length=None,
                    a_max=self.a_max,
                    max_exposure=self.max_exposure,
                    lambda_risk=self.lambda_risk,
                )
                path = self._evaluate(
                    model,
                    environment,
                    portfolio_states[seed],
                )
                portfolio_states[seed] = environment.portfolio_state()
                path["seed"] = seed
                path["fold"] = fold
                paths.append(path)
                # Le run est long : ces trois chiffres des maintenant permettent
                # d'arreter tot une politique figee ou franchement mauvaise. Le Sharpe
                # passe par le meme tableau que fold_metrics_, il ne peut donc plus
                # en diverger si la convention d'annualisation change.
                sharpe = self._paper.table(path["net_return"])["Sharpe"].iloc[0]
                print(
                    f"  seed {seed} | {time.perf_counter() - started:.0f} s "
                    f"| A moyen {path['multiplier'].mean():.3f} "
                    f"| dispersion {path['multiplier'].std():.3f} "
                    f"| sharpe {sharpe:.2f}",
                    flush=True,
                )

        self.folds_ = pd.DataFrame(fold_rows).set_index("fold")
        self.decisions_ = pd.concat(paths).sort_index()
        self.daily_returns_ = self._returns(data, splits)
        self.metrics_ = self._stats(self.daily_returns_)
        self.paper_metrics_ = self._paper.table(self.daily_returns_)

        fold_metrics, paper_folds = [], []
        for fold, dates in self.folds_.iterrows():
            block = self.daily_returns_.loc[dates["test_start"] : dates["test_end"]]
            fold_metrics.append(self._indexed(self._stats(block), fold))
            paper_folds.append(self._indexed(self._paper.table(block), fold))
        self.fold_metrics_ = pd.concat(fold_metrics)
        self.paper_fold_metrics_ = pd.concat(paper_folds)
        self.criterion_ = self._criterion()
        return self

    def mean_decisions(self):
        """Moyenne les decisions SAC entre les seeds."""
        columns = ["multiplier", "exposure", "baseline_exposure"]
        return self.decisions_.groupby(level=0)[columns].mean()

    def summary(self):
        """Affiche le critere de decision, fixe avant de regarder les resultats."""
        folds = len(self.criterion_)
        for column in self.criterion_.select_dtypes("bool"):
            print(f"{column} : {int(self.criterion_[column].sum())}/{folds} folds")
        return self.criterion_

    def _splits(self, index: pd.DatetimeIndex):
        """Decoupe en annees civiles ; les blocs de test ne se recouvrent jamais."""
        splits = []
        start = pd.Timestamp(self.first_train_start)
        while True:
            train_end_date = start + pd.DateOffset(years=self.train_years)
            test_end_date = train_end_date + pd.DateOffset(years=self.test_years)
            train_end = int(index.searchsorted(train_end_date))
            test_end = int(index.searchsorted(test_end_date))
            
            if test_end >= len(index):
                break
            train_start = int(index.searchsorted(start)) if self.rolling_train else 0
            splits.append((train_start, train_end, test_end))
            start += pd.DateOffset(years=self.test_years)
        return splits

    def _evaluate(self, model: SAC, env: ExposureEnv, portfolio_state: dict | None):
        """Fait parcourir un bloc OOS au SAC sans exploration."""
        rows = []
        observation, _ = env.reset(options=portfolio_state)
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, _, done, info = env.step(action)
            rows.append(info)
        return pd.DataFrame(rows).set_index("date")

    def _returns(
        self,
        data: pd.DataFrame,
        splits: list[tuple[int, int, int]],
    ):
        """Aligne les rendements OOS de toutes les strategies comparees."""
        returns = {
            f"SAC_seed_{seed}": self.decisions_.loc[
                self.decisions_["seed"] == seed, "net_return"
            ]
            for seed in self.seeds
        }

        oos = pd.concat(
            [data.iloc[train_end:test_end] for _, train_end, test_end in splits]
        )
        asset_return = data["price"].shift(-1).loc[oos.index] / oos["price"] - 1.0

        # La baseline historique et les references ajoutees suivent exactement la
        # meme convention : rendement du lendemain, turnover contre la veille, et
        # position initiale a plat, donc premier turnover egal a la position prise.
        for name, column in self._exposure_columns().items():
            exposure = oos[column]
            turnover = exposure.diff().abs()
            turnover.iloc[0] = abs(exposure.iloc[0])
            returns[name] = exposure * asset_return - self.cost_rate * turnover

        # Achat simple de l'actif sous-jacent, un seul cout d'entree. Le nom reste
        # neutre : l'instrument est fixe dans le notebook, pas ici.
        returns["Buy_and_hold"] = asset_return.copy()
        returns["Buy_and_hold"].iloc[0] -= self.cost_rate

        frame = pd.DataFrame(returns)
        # Portefeuille equipondere sur les seeds : c'est la courbe tracee en section 5,
        # qui ne correspondait auparavant a aucune ligne de metrics_. Attention, ce
        # n'est pas la colonne SAC de _criterion : celle-la moyenne les Sharpes, alors
        # que ce portefeuille beneficie de la diversification entre seeds et affiche
        # donc un Sharpe superieur. Les deux sont reportes, ils ne mesurent pas la
        # meme chose et _criterion reste le critere fixe avant de voir les resultats.
        frame.insert(
            0,
            "SAC_moyen",
            frame[[f"SAC_seed_{seed}" for seed in self.seeds]].mean(axis=1),
        )
        return frame

    def _exposure_columns(self):
        """Nom affiche -> colonne d'exposition, baseline historique en tete."""
        return {"Baseline": "w_base", **self.baseline_columns}

    def _opponents(self):
        """Strategies auxquelles SAC est compare, dans l'ordre d'affichage."""
        return [*self._exposure_columns(), "Buy_and_hold"]

    def _criterion(self):
        """Compare SAC a chaque adversaire, fold par fold, sur le Sharpe OOS."""
        sharpe = self.fold_metrics_["sharpe_0pct"].unstack("strategy")
        seed_columns = [c for c in sharpe.columns if c.startswith("SAC_seed")]
        # Le Sharpe est invariant a l'echelle : un multiplicateur constant reproduirait
        # celui de la baseline. A_dispersion est donc reportee a cote des comparaisons,
        # sans les conditionner : une victoire adossee a une dispersion nulle ne dit
        # rien de SAC, mais c'est au lecteur du tableau d'en juger.
        multiplier = self.decisions_.groupby(["fold", "seed"])["multiplier"]
        opponents = self._opponents()
        table = pd.DataFrame(
            {
                "SAC": sharpe[seed_columns].mean(axis=1),
                **{name: sharpe[name] for name in opponents},
                "A_moyen": multiplier.mean().groupby("fold").mean(),
                "A_dispersion": multiplier.std().groupby("fold").mean(),
            }
        )
        for name in opponents:
            table[f"SAC > {name}"] = table["SAC"] > table[name]
        return table

    @staticmethod
    def _indexed(stats: pd.DataFrame, fold: int):
        """Pose les cles fold/strategy attendues par les tableaux par fold."""
        stats = stats.copy()
        stats["fold"] = fold
        stats["strategy"] = stats.index
        return stats.set_index(["fold", "strategy"])

    def _stats(self, returns: pd.DataFrame):
        """Calcule les indicateurs financiers pour chaque colonne.

        Trois des quatre colonnes sont reprises telles quelles du tableau du
        papier, pour qu'un Sharpe ne puisse pas differer d'un tableau a l'autre.
        Seule annual_return reste propre a ce projet : elle compose, la ou E(R)
        du papier annualise arithmetiquement.
        """
        paper = self._paper.table(returns)
        equity = (1.0 + returns).cumprod()
        return pd.DataFrame(
            {
                "annual_return": equity.iloc[-1] ** (252 / len(returns)) - 1.0,
                "annual_volatility": paper["Std(R)"],
                "sharpe_0pct": paper["Sharpe"],
                # Signe negatif conserve : convention des tableaux deja publies.
                "max_drawdown": -paper["MDD"],
            }
        )
