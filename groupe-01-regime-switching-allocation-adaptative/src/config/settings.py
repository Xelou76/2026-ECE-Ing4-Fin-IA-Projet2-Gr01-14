"""
config/settings.py
==================
Centralisation de tous les hyperparamètres et configurations du projet.
Utilise Pydantic v2 pour la validation des types et la lecture depuis .env.

Corrections v3 :
  - VAEConfig : hidden_dim 128→64, latent_dim 16→8, dropout 0.2→0.3
    epochs 250→150, beta_warmup_epochs 80→40, patience 30→20
    Raison : gap train/val = 0.32 → overfitting structurel.
    8 dimensions latentes sont largement suffisantes pour 2-3 régimes.
  - DataConfig : sequence_length 60→30
    Raison : seq_len=60 réduit le nb de séquences disponibles et aggrave
    l'overfitting. 30 jours capte déjà les tendances inter-régimes.
  - HMMConfig : n_regimes 3→2 pour la robustesse
    Raison : avec 3 régimes, le régime 2 (Bull) avait proba≈1e-290 sur
    toutes les observations — le HMM n'utilisait effectivement que 2 états.
    2 régimes (bear/bull) est plus robuste et plus interprétable.
  - StrategyConfig : allocations adaptées à 2 régimes + seuil confiance 0.60
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    """Configuration pour le téléchargement et le preprocessing des données."""

    tickers: list[str] = Field(
        default=["SPY", "TLT", "GLD", "^VIX"],
        description="Tickers yfinance à télécharger",
    )
    benchmark_ticker: str = Field(default="SPY", description="Actif du backtest")

    # Période étendue pour inclure plusieurs cycles complets :
    # 2010-2019 bull market, 2020 Covid crash, 2022 bear (-20%), 2023-2024 rally
    # → la période de test (2022-2024) inclut un vrai bear market
    start_date: str = Field(default="2010-01-01", description="Début de la fenêtre")
    end_date: str = Field(default="2024-12-31", description="Fin de la fenêtre")

    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.3)

    vol_windows: list[int] = Field(default=[5, 21, 63])
    return_windows: list[int] = Field(default=[1, 5, 21])
    rsi_window: int = Field(default=14)
    bb_window: int = Field(default=20)

    # CORRECTION : séquence réduite de 60 à 30 jours.
    # Raison principale : avec seq_len=60, le nombre de séquences
    # disponibles est divisé par 2 → overfitting aggravé.
    # 30 jours capte déjà un mois calendaire de tendance, ce qui est
    # suffisant pour distinguer bull/bear à l'échelle des régimes.
    sequence_length: int = Field(
        default=30,
        description="Lookback LSTM-VAE en jours",
    )

    @field_validator("train_ratio", "val_ratio")
    @classmethod
    def ratios_sum_valid(cls, v: float) -> float:
        return v


class VAEConfig(BaseSettings):
    """Hyperparamètres du Variational Autoencoder (LSTM-VAE) v3."""

    input_dim: int = Field(default=-1, description="Déterminé dynamiquement")

    # CORRECTION : hidden_dim réduit de 128 à 64.
    # Raison : gap train/val sur recon = 0.32 → overfitting structurel.
    # 64 dimensions cachées réduisent la capacité du modèle sans perdre
    # l'expressivité nécessaire pour encoder les régimes.
    hidden_dim: int = Field(default=64, description="Dim cachée du LSTM")

    # CORRECTION : latent_dim réduit de 16 à 8.
    # Raison : 3 régimes n'ont pas besoin de 16 dimensions latentes.
    # 8 dimensions capturent largement la structure bear/bull + transition,
    # et réduisent la complexité du HMM en aval (curse of dimensionality).
    latent_dim: int = Field(
        default=8,
        description="Dimension de l'espace latent z",
    )

    num_layers: int = Field(default=2, description="Couches LSTM empilées")

    # CORRECTION : dropout augmenté de 0.2 à 0.3 pour compenser
    # la réduction de hidden_dim (plus de régularisation nécessaire).
    dropout: float = Field(default=0.3, ge=0.0, le=0.5)

    # CORRECTION : epochs réduit de 250 à 150.
    # Avec un modèle plus petit et une sequence_length réduite,
    # la convergence est plus rapide.
    epochs: int = Field(default=150)
    batch_size: int = Field(default=64)

    learning_rate: float = Field(default=5e-4)
    weight_decay: float = Field(default=1e-4)

    beta_start: float = Field(default=0.0, description="KL annealing start")
    beta_end: float = Field(default=1.0, description="KL annealing end")

    # CORRECTION : warmup réduit de 80 à 40 epochs (27% des 150 epochs).
    # Raison : avec latent_dim=8 et hidden_dim=64, le modèle converge
    # plus vite. Un warmup trop long retarde l'activation de l'early
    # stopping (qui est gelé pendant cette phase).
    beta_warmup_epochs: int = Field(
        default=40,
        description="Epochs pour atteindre beta_end",
    )
    grad_clip: float = Field(default=1.0, description="Gradient clipping")

    # CORRECTION : patience réduite de 30 à 20.
    # Avec le fix de l'early stopping (sur val_recon), la patience effective
    # est de 20 epochs POST-warmup → suffisant sans gaspiller du compute.
    early_stopping_patience: int = Field(default=20)


class HMMConfig(BaseSettings):
    """Hyperparamètres du Hidden Markov Model sur l'espace latent."""

    # CORRECTION : n_regimes réduit de 3 à 2.
    # Raison : avec 3 régimes, le régime 2 (Bull) avait une probabilité
    # de 1e-290 sur quasiment toutes les observations — le HMM n'utilisait
    # effectivement que 2 états. 2 régimes (bear/bull) est plus robuste,
    # plus interprétable, et correspond à la réalité empirique des données.
    # Une fois le VAE correctement entraîné, on pourra re-tester n_regimes=3.
    n_regimes: int = Field(
        default=2,
        description="2 régimes : bear, bull",
    )
    covariance_type: str = Field(
        default="full",
        description="Covariance full pour capturer les correlations inter-latents",
    )

    n_iter: int = Field(default=200, description="Itérations EM max")
    tol: float = Field(default=1e-5)
    n_init: int = Field(default=10, description="Restarts aléatoires")

    @field_validator("covariance_type")
    @classmethod
    def valid_cov(cls, v: str) -> str:
        allowed = {"full", "diag", "spherical", "tied"}
        if v not in allowed:
            raise ValueError(f"covariance_type doit être dans {allowed}")
        return v


class MarkovSwitchingConfig(BaseSettings):
    """Configuration du modèle de référence Hamilton (statsmodels)."""

    k_regimes: int = Field(default=2, description="2 régimes : bull/bear")
    order: int = Field(default=0, description="AR order 0 = switching in mean")
    switching_variance: bool = Field(default=True)


class StrategyConfig(BaseSettings):
    """
    Paramètres de la stratégie adaptative v3.

    Allocations adaptées à 2 régimes (n_regimes=2) :
    - Régime 0 (bear) : protection capital → bonds + cash dominant
    - Régime 1 (bull) : rendement → equity dominant

    Note : avec n_regimes=2, la clé 2 n'est plus utilisée. Elle est
    conservée pour rétro-compatibilité si on repasse à 3 régimes.

    min_confidence relevé à 0.60 pour filtrer les signaux incertains
    (l'ancien seuil de 0.55 était trop permissif avec 2 régimes).
    """

    regime_allocations: dict[int, dict[str, float]] = Field(
        default={
            0: {"equity": 0.10, "bond": 0.70, "cash": 0.20},  # bear : défensif
            1: {"equity": 0.80, "bond": 0.15, "cash": 0.05},  # bull : offensif
            2: {"equity": 0.40, "bond": 0.40, "cash": 0.20},  # transition (legacy)
        }
    )

    # Coûts de transaction : 10 bps = 0.10% par trade (réaliste ETF)
    transaction_cost_bps: float = Field(default=10.0)

    # Seuil de rebalancement : évite les trades trop petits
    rebalance_threshold: float = Field(default=0.05)

    risk_free_rate: float = Field(default=0.04, description="Taux sans risque 2024")

    # CORRECTION : seuil de confiance relevé de 0.55 à 0.60.
    # Avec 2 régimes, les probabilités postérieures sont naturellement
    # plus tranchées. Un seuil de 0.60 évite les signaux trop incertains
    # tout en restant actionnable.
    min_confidence: float = Field(
        default=0.60,
        description="Confiance min (max posterior) pour utiliser l'allocation du régime",
        ge=0.3,
        le=1.0,
    )

    # Utiliser la soft allocation (weighted) au lieu de hard Viterbi
    use_soft_allocation: bool = Field(
        default=True,
        description="Si True, allocation = Σ P(r_t=k) × alloc[k] (recommandé)",
    )


class ProjectConfig(BaseSettings):
    """Configuration maîtresse du projet."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    random_seed: int = Field(default=42)

    root_dir: Path = Field(default=Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default=Path("data/raw"))
    cache_dir: Path = Field(default=Path("data/cache"))
    model_dir: Path = Field(default=Path("models/checkpoints"))
    results_dir: Path = Field(default=Path("results"))
    figures_dir: Path = Field(default=Path("results/figures"))

    data: DataConfig = Field(default_factory=DataConfig)
    vae: VAEConfig = Field(default_factory=VAEConfig)
    hmm: HMMConfig = Field(default_factory=HMMConfig)
    markov: MarkovSwitchingConfig = Field(default_factory=MarkovSwitchingConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)

    log_level: str = Field(default="INFO")

    def create_directories(self) -> None:
        """Crée tous les répertoires nécessaires."""
        dirs = [
            self.data_dir,
            self.cache_dir,
            self.model_dir,
            self.results_dir,
            self.figures_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> ProjectConfig:
    """
    Retourne le singleton de configuration du projet.

    Returns
    -------
    ProjectConfig
        Instance unique de la configuration.
    """
    cfg = ProjectConfig()
    cfg.create_directories()
    return cfg 