# Market Regime Detection — VAE-HMM Pipeline

> Détection automatique de régimes de marché (bull / bear) via une approche hybride  
> **LSTM Variational Autoencoder + Hidden Markov Model** avec backtest causal.

---

## Vue d'ensemble

Ce projet répond à un problème concret de gestion de portefeuille : comment détecter en temps réel si le marché est dans un régime de risque élevé (bear) ou faible (bull), et adapter l'allocation en conséquence, **sans jamais utiliser d'information future** (causalité garantie).

**Pipeline en 5 étapes :**

```
Données brutes (SPY, TLT, GLD, VIX)
    ↓  Feature Engineering (64 features, 13 familles)
    ↓  LSTM-VAE — compression 64D → 8D (espace latent)
    ↓  GaussianHMM — détection de 2 régimes sur l'espace latent
    ↓  Forward Filtering — inférence causale (pas de look-ahead)
    ↓  Backtest adaptatif — allocation dynamique equity/bond/cash
```

**Résultats clés (période de test 2019–2025) :**

| Métrique | VAE-HMM | Buy & Hold | Hamilton MS |
|----------|---------|------------|-------------|
| Sharpe Ratio | 0.35 | 0.63 | 0.08 |
| CAGR | 7.5% | 15.8% | 4.4% |
| Max Drawdown | **−26.0%** | −33.7% | — |
| IC (5 jours) | **+0.045** | — | — |
| Accord Viterbi/Causal | **98.7%** | — | — |

---

## Structure du projet

```
src/
├── config/
│   ├── settings.py          # Tous les hyperparamètres (Pydantic v2)
│   └── constants.py         # Labels régimes, couleurs, noms de colonnes
├── data/
│   ├── downloader.py        # Téléchargement via yfinance
│   ├── features.py          # Feature Engineering (64 features, 13 familles)
│   └── processor.py         # Pipeline complet : download → scale → séquences
├── models/
│   ├── vae.py               # TimeSeriesVAE (LSTM-VAE, PyTorch)
│   ├── hmm.py               # RegimeHMM (GaussianHMM + inférence causale)
│   ├── markov_switching.py  # Baseline Hamilton (statsmodels)
│   └── trainer.py           # VAETrainer (KL annealing, early stopping)
├── strategy/
│   └── backtester.py        # AdaptiveStrategyBacktester (vectorisé NumPy)
├── evaluation/
│   └── comparator.py        # ModelComparator (métriques + Jobson-Korkie)
├── utils/
│   ├── metrics.py           # Sharpe, Sortino, MDD, IC, rolling Sharpe
│   ├── plotting.py          # Visualisations régimes, equity curves, latent space
│   └── seed.py              # Reproductibilité totale (seed 42)
├── tests/                   # Tests unitaires pytest
├── app.py                   # Interface Streamlit
└── main.py                  # Point d'entrée principal
```

---

## Installation

**Prérequis :** Python 3.10+, pip, optionnel : GPU CUDA pour l'entraînement

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd market_regime

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Copier la configuration (optionnel)
cp .env.example .env
```

---

## Utilisation

### Pipeline complet (entraînement + backtest)

```bash
python main.py
```

Cela exécute les 5 phases en séquence :
1. Téléchargement et feature engineering
2. Entraînement du LSTM-VAE
3. Entraînement du HMM sur l'espace latent
4. Backtest avec forward filtering
5. Génération des figures dans `results/figures/`

### Options disponibles

```bash
# Charger un checkpoint existant (passe l'entraînement)
python main.py --skip-training

# Modifier le seed de reproductibilité
python main.py --seed 123

# Tester avec plus de régimes
python main.py --n-regimes 3

# Restreindre la période de données
python main.py --start-date 2010-01-01 --end-date 2023-12-31

# Forcer le re-téléchargement des données
python main.py --force-refresh

# Utiliser la hard allocation (Viterbi) au lieu de la soft allocation
python main.py --hard-alloc
```

### Interface Streamlit

```bash
streamlit run app.py
```

Lance une interface web interactive pour visualiser les régimes, les equity curves et les KPIs.

---

## Méthode

### 1. Données et Feature Engineering

**Univers :** SPY (S&P 500), TLT (Treasuries 20 ans), GLD (Or), ^VIX (Volatilité implicite)  
**Période :** 2007–2025 (18 ans, 3 cycles complets)  
**Split :** Train 58% (2007–2018) | Val 6% (2018–2019) | Test 36% (2019–2025)

**64 features construites sur 13 familles :**

| Famille | Features | Objectif |
|---------|----------|----------|
| Log-rendements | 1j, 5j, 21j × 4 actifs | Dynamique de prix multi-horizons |
| Volatilité réalisée | 5j, 21j, 63j × 4 actifs | Niveau de risque absolu |
| Moments supérieurs | Skewness, Kurtosis glissants | Tail risk, asymétrie de distribution |
| Indicateurs techniques | RSI(14), Bollinger (largeur + position), MACD | Momentum, tendance |
| Z-score de volatilité | (vol − avg_vol) / std_vol | Régime de vol relatif |
| Drawdown courant | (prix − max_glissant) / max_glissant | Stress direct du marché |
| Autocorrélation | Corr(r_t, r_{t-k}) | Momentum vs mean-reversion |
| Corrélations inter-actifs | SPY-TLT, SPY-GLD, SPY-VIX glissantes | Flight to quality |
| Ratio vol court/long | vol_5j / vol_63j | Accélération de volatilité |
| Tendance EMA | EMA20 / EMA50 | Uptrend vs downtrend |

**Note critique :** Le RobustScaler est fitté **uniquement sur le train set** et appliqué au val et test sans refitting. Aucune fuite d'information.

### 2. LSTM-VAE

**Architecture :**

```
Input (B, 30, 64)
    → LSTM Bidirectionnel (2 couches, hidden=64, bidir → 128 dim)
    → Temporal Attention Pooling (apprend quels timesteps sont informatifs)
    → LayerNorm + Dropout(0.3)
    → fc_mu (128 → 8)  |  fc_log_var (128 → 8)
    → Reparameterization trick : z = μ + ε × exp(0.5 × log_var), ε ~ N(0,I)
    → LSTM Decoder (8 → 128 → (B, 30, 64))
```

**Perte :** ELBO = recon_loss + β × kl_loss  
**KL Annealing :** β monte de 0 à 1.5 sur les 40 premières epochs (évite le posterior collapse)  
**Early stopping :** sur val_recon (pas val_ELBO), gelé pendant le warmup

**Correction critique (bug #3) :** L'encodeur bidirectionnel extrayait incorrectement `lstm_out[:, -1, :]` pour les deux directions. La direction backward à t=T n'a vu qu'un seul pas de temps. Correction : Temporal Attention Pooling sur toutes les sorties.

**Résultats d'entraînement :**
- Best epoch : 47
- val_recon : 0.454
- val_kl : 0.013
- β_end : 1.5

### 3. HMM sur l'espace latent

Le vecteur μ de chaque séquence est passé au GaussianHMM :

```
μ ∈ ℝ⁸ (par séquence)
    → StandardScaler (fitté sur train)
    → GaussianHMM(n_components=2, covariance_type='full', n_iter=200)
    → 10 restarts, meilleur modèle retenu
    → Ordonnancement des régimes par rendement financier moyen
```

**Matrice de transition obtenue :**
- P(Bear→Bear) = 0.979 → durée espérée Bear : ~47 jours
- P(Bull→Bull) = 0.997 → durée espérée Bull : ~333 jours

**Pourquoi 2 régimes ?** Le test avec 3 régimes produisait un état avec probabilité ~10⁻²⁹⁰ sur toutes les observations — le HMM n'utilisait effectivement que 2 états. Fallback automatique implémenté.

### 4. Inférence causale (Forward Filtering)

Pour le backtest, on n'utilise **pas** Viterbi (qui est non-causal). On utilise le filtre forward :

```
α_t(k) = P(r_t=k | z_{1:t})

Récurrence : log α_t = log Σ_k [ α_{t-1}(k) × P(r_t|r_{t-1}=k) ] + log p(z_t|r_t)

Décision MAP : r_t* = argmax_k α_t(k)
```

**Accord Viterbi / Forward filtering : 98.7%** — les deux méthodes donnent quasi-systématiquement le même résultat, confirmant la robustesse de l'inférence.

### 5. Stratégie adaptative

**Allocations par régime :**

| Régime | Equity (SPY) | Obligations (TLT) | Cash |
|--------|-------------|-------------------|------|
| Bear (0) | 10% | 70% | 20% |
| Bull (1) | 80% | 15% | 5% |

**Soft allocation :** `alloc_t = Σ_k P(r_t=k) × alloc[k]`  
**Frais de transaction :** 10 bps par rebalancement  
**Seuil de dérive :** 5% (évite les micro-rebalancements)  
**Seuil de confiance :** 75% (sous ce seuil → allocation Bear par défaut)

---

## Corrections critiques appliquées

| # | Problème | Correction |
|---|----------|------------|
| 1 | Early stopping sur val_ELBO | Changé pour val_recon (ELBO monte artificiellement pendant le warmup) |
| 2 | Early stopping actif pendant KL warmup | Gel du compteur pendant beta_warmup_epochs |
| 3 | Bug encodeur LSTM bidirectionnel | lstm_out[:,-1,hidden:] → Temporal Attention Pooling (voir vae.py) |
| 4 | Checkpoint sur val_ELBO | Changé pour val_recon |
| 5 | 3 régimes HMM instables | Fallback automatique vers 2 régimes si état dégénéré détecté |

---

## Reproductibilité

Tous les seeds sont fixés via `utils.seed.set_all_seeds(42)` :
- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- `PYTHONHASHSEED=42`

Pour reproduire exactement les résultats : `python main.py --seed 42`

---

## Tests

```bash
# Tous les tests
pytest tests/ -v

# Avec coverage
pytest tests/ -v --cov=. --cov-report=html

# Test spécifique
pytest tests/test_vae.py -v
pytest tests/test_hmm.py -v
pytest tests/test_strategy.py -v
```

---

## Limites connues

| Limite | Impact | Solution envisagée |
|--------|--------|-------------------|
| Retard ~3 mois sur les débuts de bear | MDD sous-optimal sur 2022 | Réduire la contrainte de persistance (p_00) |
| Overfitting VAE partiel | Gap train/val persistant | Augmentation de données, dropout plus agressif |
| Sensibilité à β_end | Performances varient selon β | Grid search Optuna |
| Data snooping potentiel | Test set visualisé pendant le dev | Holdout final strict sur les prochaines versions |
| Généralisation non validée | Calibré uniquement sur SPY | Extension multi-actifs, marchés émergents |

---

## Prochaines étapes (Roadmap)

- [ ] Grid search β via Optuna
- [ ] Extension à 3 régimes stables (une fois le VAE optimisé)
- [ ] Modèle online avec mise à jour incrémentale (Incremental EM)
- [ ] Monitoring PSI en production (détection de dérive de distribution)
- [ ] Extension multi-actifs (actions européennes, EM, commodities)
- [ ] Test de significativité Jobson-Korkie complet

---

## Dépendances principales

| Package | Version | Usage |
|---------|---------|-------|
| torch | ≥2.0 | LSTM-VAE |
| hmmlearn | ≥0.3 | GaussianHMM |
| yfinance | ≥0.2 | Téléchargement données |
| ta | ≥0.11 | Indicateurs techniques |
| pydantic-settings | ≥2.0 | Configuration typée |
| statsmodels | ≥0.14 | Baseline Hamilton |
| scikit-learn | ≥1.3 | StandardScaler, métriques |
| scipy | ≥1.11 | Tests statistiques, Spearman IC |
| streamlit | ≥1.30 | Interface web |
| loguru | ≥0.7 | Logging structuré |

---

## Auteur

Maisonnave Gabriel 
Raph couvert 
Aurele gsquet  
Seed : 42 | Période : 2007–2025 | Framework : PyTorch + hmmlearn