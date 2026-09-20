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
    ├── baseline.py        stratégies de référence, celle du projet et celles du papier
    ├── environment.py      environnement Gymnasium pour SAC
    ├── scaling.py          normalisation des features
    ├── metrics.py          indicateurs de performance du papier de référence
    └── runner.py            validation walk-forward
```

### Stratégies comparées

Quatre stratégies d'exposition servent de point de comparaison, toutes construites
sur la même ossature : un signal directionnel dans `[-1, 1]`, mis à l'échelle pour
viser 10 % de volatilité, puis borné à `w_max`. Seul le signal les distingue, ce
qui est la condition pour qu'un écart de performance s'attribue au signal et non à
un budget de risque différent.

| Stratégie | Signal |
|---|---|
| `TrendVolatilityBaseline` | signe du rendement cumulé sur 63 séances, la baseline du projet |
| `LongOnly` | constante à 1, la baseline *Long Only* du papier |
| `SignReturn` | signe du rendement cumulé sur 252 séances, *Sign(R)*, équation 10 |
| `MACDSignal` | moyenne des réponses MACD sur trois paires d'échelles, équations 3, 11 et 12 |

`LongOnly` n'est pas `Buy_and_hold`. La première vise une volatilité cible et se
redimensionne chaque jour, la seconde détient une unité de l'indice sans jamais se
rebalancer. Leur écart isole l'apport du ciblage de volatilité.

Trois points du MACD ne se lisent pas dans le papier seul. Ils sont tranchés par sa
source, Lim, Zohren et Roberts (2019)
([arXiv:1904.04912](https://arxiv.org/abs/1904.04912)), et par l'implémentation de
référence de ce groupe. Le code les documente.

- **Moyenne, pas somme.** L'équation 12 écrit une somme là où le texte qui la
  précède dit « average them ». L'équation 8 de la source porte la même
  contradiction, et son implémentation de référence divise par le nombre de paires.
  La lettre de l'équation se réfute d'ailleurs seule : une somme de trois quantités
  déjà normalisées passe 71 % du temps au-delà de la racine de deux, où la fonction
  de réponse redescend vers zéro.
- **La moyenne porte sur les tendances, pas sur les réponses.** La fonction de
  réponse s'applique une fois, au signal agrégé. C'est l'ordre des équations 7 et 8
  de la source.
- **Demi-vie.** Le papier ne donne pas la conversion de l'échelle `S`. La source la
  donne : `HL = log(0.5) / log(1 - 1/S)`.

Un point s'écarte volontairement de l'implémentation de référence. Celle-ci remplit
l'échauffement des écarts-types glissants par backfill, ce qui y fait entrer une
valeur future. Les valeurs manquantes sont ici laissées vides, et le notebook retire
ces lignes.

### Mesure de la performance

Les résultats sont reportés avec les neuf indicateurs de la section 4.4 de Zhang,
Zohren et Roberts (2019), *Deep Reinforcement Learning for Trading*
([arXiv:1911.10107](https://arxiv.org/abs/1911.10107)) : `E(R)`, `Std(R)`, `DD`,
`Sharpe`, `Sortino`, `MDD`, `Calmar`, `% +ve Returns` et `Ave. P / Ave. L`.

Deux conventions méritent attention. `E(R)` annualise arithmétiquement quand
`annual_return` compose. Et `DD` est la semi-déviation, la racine du moment
d'ordre deux de `min(0, r)` sur toutes les séances, **pas** ce que décrit la prose
du papier. Sa phrase dit « standard deviation of trade returns that are negative »,
mais son implémentation de référence appelle `empyrical.downside_risk`, qui est la
semi-déviation, et le rapport `DD/Vol` publié par sa source le confirme. La lecture
littérale reste accessible via `PaperMetrics(downside='negative_std')`, et un
Sortino ne se compare qu'à un Sortino calculé de la même façon.

`Calmar` garde un numérateur arithmétique, ce qui reproduit exactement les tables
publiées par la source. `empyrical.calmar_ratio` compose le sien et donne un
chiffre différent.

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
    ├── baseline.py        reference strategies, this project's and the paper's
    ├── environment.py      Gymnasium environment for SAC
    ├── scaling.py          feature normalization
    ├── metrics.py          reference paper performance metrics
    └── runner.py            walk-forward validation
```

### Strategies compared

Four exposure strategies serve as benchmarks, all built on the same skeleton: a
directional signal in `[-1, 1]`, scaled to target 10% volatility, then capped at
`w_max`. Only the signal differs between them, which is what makes a performance
gap attributable to the signal rather than to a different risk budget.

| Strategy | Signal |
|---|---|
| `TrendVolatilityBaseline` | sign of the 63-day cumulative return, this project's baseline |
| `LongOnly` | constant 1, the paper's *Long Only* baseline |
| `SignReturn` | sign of the 252-day cumulative return, *Sign(R)*, equation 10 |
| `MACDSignal` | mean MACD response across three timescale pairs, equations 3, 11 and 12 |

`LongOnly` is not `Buy_and_hold`. The first targets a volatility level and resizes
daily, the second holds one unit of the index and never rebalances. The gap
between them isolates what volatility targeting contributes.

Three points of the MACD cannot be read from the paper alone. They are settled by
its source, Lim, Zohren and Roberts (2019)
([arXiv:1904.04912](https://arxiv.org/abs/1904.04912)), and by that group's
reference implementation. The code documents them.

- **Mean, not sum.** Equation 12 writes a sum where the text preceding it says
  "average them". Equation 8 of the source carries the same contradiction, and its
  reference implementation divides by the number of pairs. The literal equation also
  refutes itself: a sum of three already-normalised quantities spends 71% of the time
  beyond the square root of two, where the response function decays back to zero.
- **The mean is taken over trend estimates, not over responses.** The response
  function is applied once, to the aggregated signal. That is the order of equations
  7 and 8 in the source.
- **Halflife.** The paper does not give the conversion from timescale `S`. The source
  does: `HL = log(0.5) / log(1 - 1/S)`.

One point deliberately departs from the reference implementation. It backfills the
warmup of its rolling standard deviations, which lets a future value in. Missing
values are left empty here, and the notebook drops those rows.

### Measuring performance

Results are reported with the nine metrics from section 4.4 of Zhang, Zohren and
Roberts (2019), *Deep Reinforcement Learning for Trading*
([arXiv:1911.10107](https://arxiv.org/abs/1911.10107)): `E(R)`, `Std(R)`, `DD`,
`Sharpe`, `Sortino`, `MDD`, `Calmar`, `% +ve Returns` and `Ave. P / Ave. L`.

Two conventions deserve attention. `E(R)` annualizes arithmetically where
`annual_return` compounds. And `DD` is the semi-deviation, the root of the second
moment of `min(0, r)` over all sessions, **not** what the paper's prose describes.
Its sentence says "standard deviation of trade returns that are negative", but its
reference implementation calls `empyrical.downside_risk`, which is the
semi-deviation, and the `DD/Vol` ratio published by its source confirms it. The
literal reading stays available through `PaperMetrics(downside='negative_std')`,
and a Sortino only compares to a Sortino computed the same way.

`Calmar` keeps an arithmetic numerator, which reproduces the source's published
tables exactly. `empyrical.calmar_ratio` compounds its own and gives a different
number.

### Running the project

```bash
git clone https://github.com/AdemmBr/SAC_control.git
cd SAC_control
pip install -r requirements.txt
jupyter lab SAC_control_project.ipynb
```
