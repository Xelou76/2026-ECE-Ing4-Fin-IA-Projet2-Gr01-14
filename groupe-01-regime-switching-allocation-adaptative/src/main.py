"""
main.py
=======
Point d'entrée principal du projet VAE-HMM Market Regime Detection.

Orchestration complète du pipeline en 6 phases :
  Phase 1 — Setup : seeds, logging, configuration
  Phase 2 — Données : téléchargement, feature engineering, splits
  Phase 3 — VAE : entraînement et extraction des représentations latentes
  Phase 4 — HMM : fitting sur l'espace latent, baseline Hamilton
  Phase 5 — Backtest : stratégie adaptative, Buy-and-Hold, Hamilton
  Phase 6 — Évaluation : métriques, comparaison, figures, rapport

Usage
-----
    # Lancement avec la configuration par défaut
    python main.py

    # Ou avec des overrides de configuration
    python main.py --n_regimes 2 --latent_dim 16 --seed 123

Architecture du code
--------------------
Ce fichier n'implémente aucune logique métier. Il orchestre uniquement
les modules dédiés dans data/, models/, strategy/, evaluation/, utils/.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)
logger.add(
    "artifacts/pipeline.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="10 MB",
)

# ---------------------------------------------------------------------------
# Imports projet
# ---------------------------------------------------------------------------

from config.settings import ProjectConfig, DEFAULT_CONFIG
from utils.seeds import set_all_seeds
from strategy.adaptive_strategy import AdaptiveStrategyBacktester
from evaluation.metrics import ModelComparator, PerformanceReport, RegimeMetrics
from utils.plotting import RegimePlotter


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: ProjectConfig) -> None:
    """
    Exécute l'intégralité du pipeline VAE-HMM.

    Parameters
    ----------
    cfg : ProjectConfig
        Configuration globale du projet.
    """

    # ======================================================================
    # PHASE 1 : Setup
    # ======================================================================
    logger.info("=" * 60)
    logger.info(f"  {cfg.project_name}")
    logger.info("=" * 60)
    logger.info("PHASE 1 — Setup & Configuration")

    set_all_seeds(cfg.vae.seed)

    Path("artifacts/figures").mkdir(parents=True, exist_ok=True)
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts/data").mkdir(parents=True, exist_ok=True)
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # PHASE 2 : Données
    # ======================================================================
    logger.info("\nPHASE 2 — Téléchargement & Feature Engineering")

    try:
        from data.processor import MarketDataProcessor
        processor = MarketDataProcessor(cfg.data)
        data_dict = processor.run()

        price_series = data_dict["price"]           # pd.Series : cours de clôture SPY
        returns_series = data_dict["returns"]       # pd.Series : log-returns
        features_scaled = data_dict["features"]     # np.ndarray : features normalisées
        feature_dates = data_dict["dates"]          # pd.DatetimeIndex

        # Splits train / val / test
        splits = processor.get_splits(features_scaled, feature_dates)
        X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
        dates_train, dates_val, dates_test = (
            splits["dates_train"], splits["dates_val"], splits["dates_test"]
        )
        returns_test = returns_series.loc[dates_test]
        price_test = price_series.loc[dates_test]

    except ImportError:
        logger.warning(
            "Module data.processor non trouvé. "
            "Génération de données synthétiques pour la démonstration."
        )
        price_series, returns_series, features_scaled, feature_dates = (
            _generate_synthetic_data(cfg)
        )
        n = len(features_scaled)
        n_train = int(n * cfg.data.train_ratio)
        n_val = int(n * cfg.data.val_ratio)

        X_train = features_scaled[:n_train]
        X_val = features_scaled[n_train:n_train + n_val]
        X_test = features_scaled[n_train + n_val:]

        dates_train = feature_dates[:n_train]
        dates_val = feature_dates[n_train:n_train + n_val]
        dates_test = feature_dates[n_train + n_val:]

        returns_test = returns_series.iloc[n_train + n_val:]
        price_test = price_series.iloc[n_train + n_val:]

    logger.info(
        f"  Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)} observations"
    )

    # ======================================================================
    # PHASE 3 : VAE
    # ======================================================================
    logger.info("\nPHASE 3 — Entraînement du VAE (LSTM-VAE)")

    try:
        from models.vae import TimeSeriesVAE
        from models.trainer import VAETrainer
        from config.settings import VAEConfig

        vae_cfg = replace(cfg.vae, input_dim=X_train.shape[-1])
        vae = TimeSeriesVAE(vae_cfg)
        trainer = VAETrainer(vae, vae_cfg)
        history = trainer.fit(X_train, X_val)

        # Extraction des représentations latentes (μ du VAE)
        latent_train = trainer.encode(X_train)  # (N_train, latent_dim)
        latent_val = trainer.encode(X_val)
        latent_test = trainer.encode(X_test)

        vae.save(Path("artifacts/models/vae.pt"))

    except (ImportError, Exception) as e:
        logger.warning(f"VAE indisponible ({e}). Utilisation de features brutes comme proxy latent.")
        # Fallback : PCA sur les features comme proxy d'espace latent
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(features_scaled)

        pca = PCA(n_components=min(cfg.vae.latent_dim, X_train.shape[1] - 1))
        n_train = len(X_train)
        n_val = len(X_val)

        latent_all = pca.fit_transform(X_all_scaled)
        latent_train = latent_all[:n_train]
        latent_val = latent_all[n_train:n_train + n_val]
        latent_test = latent_all[n_train + n_val:]

        logger.info(
            f"  PCA fallback — variance expliquée : "
            f"{pca.explained_variance_ratio_.sum() * 100:.1f}%"
        )

    # ======================================================================
    # PHASE 4 : HMM & Baseline
    # ======================================================================
    logger.info("\nPHASE 4 — HMM sur espace latent & Baseline Hamilton")

    # --- VAE-HMM ---
    from models.hmm import RegimeHMM

    hmm = RegimeHMM(cfg.hmm)
    hmm.fit(latent_train)
    hmm.print_diagnostics(latent_train)

    regimes_test = hmm.predict(latent_test)
    regime_proba_test = hmm.predict_proba(latent_test)
    transition_matrix = hmm.get_transition_matrix()

    hmm.save(Path("artifacts/models/hmm.pkl"))

    # --- Baseline Markov-Switching (Hamilton) ---
    try:
        from models.markov_switching import MarkovSwitchingBaseline

        baseline = MarkovSwitchingBaseline(cfg.markov_switching)
        baseline.fit(returns_series.iloc[:len(X_train) + len(X_val)])
        baseline_regimes_test = baseline.predict(returns_test)
        baseline_available = True
        logger.success("Baseline Hamilton ajusté avec succès.")

    except Exception as e:
        logger.warning(f"Baseline Hamilton indisponible ({e}). Régimes synthétiques.")
        baseline_regimes_test = (regimes_test > 0).astype(int)
        baseline_available = False

    # ======================================================================
    # PHASE 5 : Backtest
    # ======================================================================
    logger.info("\nPHASE 5 — Backtests des stratégies")

    backtester = AdaptiveStrategyBacktester(cfg.strategy)
    reporter = PerformanceReport(cfg.evaluation)

    # 1. VAE-HMM Adaptive
    result_vae = backtester.run(
        returns=returns_test,
        regimes=regimes_test,
        strategy_name="VAE-HMM Adaptive",
        rolling_window=cfg.evaluation.rolling_window,
        risk_free_rate=cfg.strategy.risk_free_rate,
    )
    result_vae = reporter.generate(result_vae)
    reporter.print_summary(result_vae)

    # 2. Buy-and-Hold (benchmark)
    result_bah = backtester.run_buy_and_hold(
        returns=returns_test,
        risk_free_rate=cfg.strategy.risk_free_rate,
    )
    result_bah = reporter.generate(result_bah)

    # 3. Hamilton Markov-Switching
    result_hamilton = backtester.run_markov_baseline(
        returns=returns_test,
        baseline_regimes=baseline_regimes_test,
        risk_free_rate=cfg.strategy.risk_free_rate,
    )
    result_hamilton = reporter.generate(result_hamilton)

    all_results = [result_vae, result_hamilton, result_bah]

    # ======================================================================
    # PHASE 6 : Évaluation & Visualisations
    # ======================================================================
    logger.info("\nPHASE 6 — Évaluation & Génération des figures")

    # --- Comparaison des modèles ---
    comparator = ModelComparator(cfg)
    for res in all_results:
        comparator.add_result(res)
    comparator.print_comparison()

    # Test de significativité Sharpe
    comparator.sharpe_significance_test("VAE-HMM Adaptive", "Buy & Hold")
    if baseline_available:
        comparator.sharpe_significance_test(
            "VAE-HMM Adaptive", "Hamilton Markov-Switching"
        )

    # --- Métriques de régimes ---
    regime_metrics = RegimeMetrics(cfg)
    regime_stats = regime_metrics.regime_conditional_stats(returns_test, regimes_test)
    logger.info(f"\n  Stats conditionnelles par régime (test):\n{regime_stats.to_string()}\n")

    ic = regime_metrics.information_coefficient(regimes_test, returns_test, horizon=1)
    ic_5d = regime_metrics.information_coefficient(regimes_test, returns_test, horizon=5)

    empirical_tm = regime_metrics.transition_matrix_empirical(regimes_test)
    logger.info(f"\n  Matrice de transition empirique (test):\n{empirical_tm.to_string()}\n")

    # --- Sauvegarde du tableau comparatif ---
    summary_df = comparator.summary_table()
    summary_path = Path("artifacts/reports/performance_comparison.csv")
    summary_df.to_csv(summary_path)
    logger.info(f"  Tableau comparatif sauvegardé : {summary_path}")

    # --- Visualisations ---
    plotter = RegimePlotter(cfg, dark_mode=True)

    logger.info("  Génération des figures...")

    plotter.plot_regimes_on_price(
        price=price_test,
        regimes=regimes_test,
        show_regime_proba=regime_proba_test,
        save="regimes_on_price.png",
    )

    plotter.plot_equity_curves(all_results, save="equity_curves.png")

    plotter.plot_transition_matrix(
        transition_matrix, model_name="VAE-HMM", save="transition_matrix_vae.png"
    )

    plotter.plot_regime_distribution(regimes_test, save="regime_distribution.png")

    plotter.plot_rolling_sharpe(all_results, save="rolling_sharpe.png")

    plotter.plot_drawdowns(all_results, save="drawdowns.png")

    plotter.plot_monthly_returns_heatmap(result_vae, save="monthly_returns_vae.png")

    plotter.plot_regime_conditional_returns(
        returns_test, regimes_test, save="regime_conditional_returns.png"
    )

    plotter.plot_latent_space(
        latent_test, regimes_test, method="pca", save="latent_space_pca.png"
    )

    # Dashboard final
    plotter.plot_full_dashboard(
        price=price_test,
        regimes_vae=regimes_test,
        results=all_results,
        transition_matrix=transition_matrix,
        regimes_baseline=baseline_regimes_test,
        save="full_dashboard.png",
    )

    logger.success("\n✅ Pipeline VAE-HMM complet — tous les artefacts générés.")
    logger.info(f"   Figures : artifacts/figures/")
    logger.info(f"   Rapports : artifacts/reports/")
    logger.info(f"   Modèles  : artifacts/models/")


# ---------------------------------------------------------------------------
# Synthetic data fallback (démo sans yfinance)
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    cfg: ProjectConfig,
    n_days: int = 2000,
) -> tuple:
    """
    Génère des données financières synthétiques pour la démonstration.

    Simule 3 régimes : bull (haute dérive, basse vol), transition, bear.
    Utilise un processus HMM-GMM simplifié comme vérité terrain.

    Parameters
    ----------
    cfg : ProjectConfig
    n_days : int
        Nombre de jours de simulation.

    Returns
    -------
    tuple : (price, returns, features, dates)
    """
    rng = np.random.default_rng(cfg.vae.seed)

    # Paramètres des 3 régimes : (mu_daily, sigma_daily, durée_moy)
    regime_params = [
        (0.0005, 0.006, 120),   # R0 : bull / low-vol
        (0.0001, 0.012, 40),    # R1 : transition
        (-0.0008, 0.022, 60),   # R2 : bear / high-vol
    ]

    # Simulation des régimes via chaîne de Markov
    transition_probs = [
        [0.98, 0.015, 0.005],
        [0.04, 0.92, 0.04],
        [0.01, 0.04, 0.95],
    ]
    A = np.array(transition_probs)

    regimes = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regimes[t] = rng.choice(3, p=A[regimes[t - 1]])

    # Génération des returns
    returns = np.zeros(n_days)
    for t in range(n_days):
        k = regimes[t]
        mu, sigma, _ = regime_params[k]
        returns[t] = rng.normal(mu, sigma)

    # Prix
    price_vals = 100 * np.exp(np.cumsum(returns))

    # Features : returns + volatilités glissantes + RSI simplifié
    r_series = pd.Series(returns)
    features_list = [r_series]
    for w in [5, 10, 21, 63]:
        features_list.append(r_series.rolling(w).std().fillna(method="bfill"))
    for w in [5, 21]:
        features_list.append(r_series.rolling(w).mean().fillna(method="bfill"))
    # Momentum
    for w in [10, 21, 63]:
        features_list.append(r_series.rolling(w).sum().fillna(method="bfill"))

    features = np.column_stack([f.values for f in features_list])

    # Normalisation
    from sklearn.preprocessing import StandardScaler
    features = StandardScaler().fit_transform(features)

    # Dates
    dates = pd.date_range(start=cfg.data.start_date, periods=n_days, freq="B")

    price_series = pd.Series(price_vals, index=dates, name=cfg.data.benchmark)
    returns_series = pd.Series(returns, index=dates, name="returns")

    logger.info(
        f"  Données synthétiques générées : {n_days} jours, "
        f"{features.shape[1]} features, 3 régimes HMM."
    )
    return price_series, returns_series, features, pd.DatetimeIndex(dates)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="VAE-HMM Market Regime Detection — Pipeline complet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_regimes", type=int, default=3,
                        help="Nombre de régimes HMM")
    parser.add_argument("--latent_dim", type=int, default=8,
                        help="Dimension de l'espace latent VAE")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed pour la reproductibilité")
    parser.add_argument("--dark_mode", action="store_true", default=True,
                        help="Thème sombre pour les figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override de la configuration par défaut
    cfg = replace(
        DEFAULT_CONFIG,
        hmm=replace(DEFAULT_CONFIG.hmm, n_regimes=args.n_regimes),
        vae=replace(DEFAULT_CONFIG.vae, latent_dim=args.latent_dim, seed=args.seed),
    )

    run_pipeline(cfg)