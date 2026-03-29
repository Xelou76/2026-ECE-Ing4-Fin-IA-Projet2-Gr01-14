# Market Regime Detection — VAE-HMM

> Détection automatique de régimes de marché (bull/bear, haute/basse volatilité)  
> via une approche hybride **Variational Autoencoder + Hidden Markov Model**.

---

## Architecture

```
market_regime/
├── config/           # Hyperparamètres centralisés (Pydantic)
│   ├── settings.py   # ProjectConfig, DataConfig, VAEConfig, HMMConfig...
│   └── constants.py  # Labels régimes, couleurs, noms de colonnes
├── data/             # Ingestion & Feature Engineering
│   ├── downloader.py # MarketDataDownloader (yfinance)
│   ├── features.py   # FeatureEngineer (rendements, vol, RSI, BB...)
│   └── processor.py  # MarketDataProcessor (pipeline complet)
├── models/           # Modèles probabilistes
│   ├── vae.py        # TimeSeriesVAE (LSTM-VAE, PyTorch)
│   ├── hmm.py        # RegimeHMM (hmmlearn)
│   ├── markov_switching.py  # MarkovSwitchingBaseline (statsmodels)
│   └── trainer.py    # VAETrainer (boucle d'entraînement + KL annealing)
├── strategy/         # Stratégie adaptative & Backtest
│   ├── signals.py    # RegimeSignalGenerator
│   ├── allocation.py # AllocationEngine
│   └── backtester.py # AdaptiveStrategyBacktester (vectorbt)
├── evaluation/       # Métriques & Comparaison
│   ├── comparator.py # ModelComparator
│   └── reporter.py   # ReportGenerator
├── utils/            # Utilitaires transversaux
│   ├── seed.py       # set_all_seeds() — reproductibilité totale
│   ├── metrics.py    # Calcul Sharpe, Sortino, Drawdown...
│   └── plotting.py   # RegimePlotter (visualisations régimes sur prix)
├── notebooks/        # Exploration interactive (non-production)
├── tests/            # Tests unitaires (pytest)
└── main.py           # Pipeline principal
```

## Installation

```bash
git clone <repo>
cd market_regime

# Environnement virtuel
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dépendances
pip install -r requirements.txt

# Config locale
cp .env.example .env
```

## Exécution

```bash
# Pipeline complet (entraînement + backtest)
python main.py

# Override des paramètres
python main.py --seed 123 --n-regimes 4 --start-date 2010-01-01

# Chargement d'un checkpoint existant
python main.py --skip-training
```

## Méthode

### 1. Feature Engineering
À partir des prix journaliers de `SPY`, `TLT`, `GLD`, `VIX` :
- Log-rendements (1j, 5j, 21j)
- Volatilité réalisée glissante (5j, 21j, 63j)
- RSI(14), Bandes de Bollinger (largeur + position)
- Tendance (EMA50 vs EMA200)

### 2. LSTM-VAE
Encode des séquences temporelles (30 jours) dans un espace latent 8D.
- **Encoder** : LSTM bidirectionnel → μ, log σ² (reparameterization trick)
- **Decoder** : LSTM → reconstruction de la séquence
- **KL Annealing** : β croît de 0 → 1 sur 50 epochs (évite posterior collapse)

### 3. HMM sur l'espace latent
`hmmlearn.GaussianHMM` avec `n_regimes=3` et matrice de covariance `full`
ajusté sur les représentations latentes de train → détection des régimes.

### 4. Baseline : Markov-Switching (Hamilton)
`statsmodels.tsa.regime_switching.markov_switching` avec 2 régimes,
ajusté directement sur les rendements du SPY.

### 5. Stratégie Adaptative
| Régime détecté | Actions (SPY) | Obligations (TLT) | Cash |
|---|---|---|---|
| Bear / Haute Vol | 10% | 70% | 20% |
| Transition | 40% | 40% | 20% |
| Bull / Basse Vol | 80% | 15% | 5% |

## Métriques de comparaison
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown, Volatilité annualisée
- Total Return, Win Rate

## Reproductibilité
Tous les seeds sont fixés via `utils.seed.set_all_seeds(seed)` :
Python `random`, NumPy, PyTorch CPU/CUDA, `PYTHONHASHSEED`.

## Tests

```bash
pytest tests/ -v --cov=. --cov-report=html
```
