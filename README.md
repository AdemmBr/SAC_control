# SAC Control

**[Français](#français)** · **[English](#english)**

---

## Français

Contrôleur d'exposition entraîné par reinforcement learning (Soft Actor-Critic, via stable-baselines3) sur l'indice S&P 500 (`^GSPC`).

Une baseline simple (tendance sur 63 jours, ciblage de volatilité à 10%) fixe une exposition de référence `w_base`. L'agent SAC apprend un multiplicateur `A_t` dans `[0, A_max]` appliqué à cette baseline : `A_t = 1` reproduit la baseline, en dessous il l'atténue, au-dessus il l'amplifie. L'exposition finale est `w_t = clip(A_t * w_base_t, ±w_max)`.

La récompense est le P&L net moins le coût de transaction (turnover) moins une pénalité de risque optionnelle.

### Ce que l'agent observe

Sept features de marché calculées à partir des rendements : momentum 21 et 63 jours, return z-score, régime de volatilité, ratio de volatilité court terme / long terme, ratio de volatilité à la baisse, et le drawdown de marché.

### Organisation

```
.
├── SAC_control_project.ipynb   notebook principal (données, entraînement, évaluation)
├── requirements.txt
└── utils/
    ├── market_data.py    chargement des prix via yfinance
    ├── features.py       features observées par l'agent
    ├── baseline.py        stratégie de référence (tendance + ciblage de volatilité)
    ├── environment.py      environnement Gymnasium pour SAC
    ├── scaling.py          normalisation des features
    └── runner.py            validation walk-forward
```

### Lancer le projet

```bash
git clone https://github.com/AdemmBr/SAC_control.git
cd SAC_control
pip install -r requirements.txt
jupyter lab SAC_control_project.ipynb
```

---

## English

An exposure controller trained with reinforcement learning (Soft Actor-Critic, via stable-baselines3) on the S&P 500 index (`^GSPC`).

A simple baseline (63-day trend, 10% volatility target) sets a reference exposure `w_base`. The SAC agent learns a multiplier `A_t` in `[0, A_max]` applied to that baseline: `A_t = 1` reproduces the baseline, below that it dampens it, above it amplifies it. Final exposure is `w_t = clip(A_t * w_base_t, ±w_max)`.

The reward is net P&L minus transaction cost (turnover) minus an optional risk penalty.

### What the agent observes

Seven market features computed from returns: 21 and 63-day momentum, return z-score, volatility regime, short/long volatility ratio, downside volatility ratio, and market drawdown.

### Layout

```
.
├── SAC_control_project.ipynb   main notebook (data, training, evaluation)
├── requirements.txt
└── utils/
    ├── market_data.py    price loading via yfinance
    ├── features.py       features the agent observes
    ├── baseline.py        reference strategy (trend + volatility targeting)
    ├── environment.py      Gymnasium environment for SAC
    ├── scaling.py          feature normalization
    └── runner.py            walk-forward validation
```

### Running the project

```bash
git clone https://github.com/AdemmBr/SAC_control.git
cd SAC_control
pip install -r requirements.txt
jupyter lab SAC_control_project.ipynb
```
