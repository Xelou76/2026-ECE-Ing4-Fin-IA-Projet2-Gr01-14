"""
main.py
=======
Point d'entrée principal — pipeline VAE-HMM complet en 5 phases.

Usage
-----
    python main.py                              # pipeline complet
    python main.py --skip-training              # charge checkpoint existant
    python main.py --seed 123 --n-regimes 2    # override paramètres
    python main.py --start-date 2010-01-01      # dates personnalisées
    python main.py --soft-alloc                 # active la soft allocation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Market Regime Detection via VAE-HMM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-regimes", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    # [NOUVEAU] Active la soft allocation (pondérée par les probabilités)
    parser.add_argument(
        "--soft-alloc",
        action="store_true",
        default=True,
        help="Utilise la soft allocation forward-filtered (recommandé)",
    )
    # [NOUVEAU] Désactive la soft allocation pour utiliser Viterbi hard
    parser.add_argument(
        "--hard-alloc",
        action="store_true",
        default=False,
        help="Utilise la hard allocation Viterbi (moins recommandé)",
    )
    return parser.parse_args()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure loguru."""
    logger.remove()
    logger.add(
        sys.stderr, level=log_level, colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    Path("results").mkdir(exist_ok=True)
    logger.add(
        "results/run_{time:YYYY-MM-DD_HH-mm-ss}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Helper : soft allocation sans modifier le backtester
# ---------------------------------------------------------------------------

def compute_soft_regimes(
    regime_hmm,
    latent_test: np.ndarray,
    regime_allocations: dict,
    min_confidence: float = 0.55,
) -> np.ndarray:
    """
    Convertit les probabilités forward en régimes "effectifs" pour le backtester.

    Stratégie : si la confiance max(P(r_t=k)) dépasse min_confidence,
    on utilise le régime argmax. Sinon, on replie vers le régime intermédiaire
    (régime 1 = transition) pour ne pas prendre de risque inutile.

    C'est un compromis entre soft allocation (idéale mais nécessite de modifier
    le backtester) et hard allocation (simple mais sensible au bruit).

    Parameters
    ----------
    regime_hmm : RegimeHMM
        Modèle HMM ajusté (v2, avec get_forward_proba).
    latent_test : np.ndarray
        Représentations latentes du test set.
    regime_allocations : dict
        Dictionnaire des allocations par régime.
    min_confidence : float
        Seuil de confiance minimale. En dessous → régime prudent (1).

    Returns
    -------
    np.ndarray
        Séquence de régimes filtrée par confiance — shape (N,).
    """
    n_regimes = len(regime_allocations)
    # Régime par défaut en cas d'incertitude : le milieu (transition)
    fallback_regime = n_regimes // 2

    # Probabilités causales P(r_t | z_{1:t}) via forward filtering
    forward_proba = regime_hmm.predict_proba_causal(latent_test)  # (N, K)

    # Argmax des probabilités forward (causal, pas look-ahead)
    raw_regimes = np.argmax(forward_proba, axis=1)  # (N,)

    # Confiance = probabilité du régime prédit
    confidence = forward_proba.max(axis=1)  # (N,)

    # Filtre : régimes incertains → fallback
    filtered_regimes = np.where(
        confidence >= min_confidence,
        raw_regimes,
        fallback_regime,
    )

    # Log du filtering
    n_filtered = (confidence < min_confidence).sum()
    pct_filtered = 100.0 * n_filtered / len(confidence)
    logger.info(
        f"  Filtre de confiance (seuil={min_confidence:.0%}) : "
        f"{n_filtered}/{len(confidence)} jours repliés ({pct_filtered:.1f}%) "
        f"→ régime {fallback_regime} (transition)"
    )

    return filtered_regimes


def main() -> int:
    args = parse_args()
    t_start = time.perf_counter()

    # Soft alloc activée par défaut, sauf si --hard-alloc explicitement fourni
    use_soft_alloc = not args.hard_alloc

    # ------------------------------------------------------------------
    # 1. Config & Seeds
    # ------------------------------------------------------------------
    from config.settings import get_settings
    from utils.seed import set_all_seeds

    cfg = get_settings()
    if args.tickers:
        cfg.data.tickers = args.tickers
    if args.start_date:
        cfg.data.start_date = args.start_date
    if args.end_date:
        cfg.data.end_date = args.end_date
    if args.n_regimes is not None:
        cfg.hmm.n_regimes = args.n_regimes

    setup_logging(cfg.log_level)
    set_all_seeds(args.seed)

    logger.info("=" * 60)
    logger.info("  Market Regime Detection — VAE-HMM Pipeline v2")
    logger.info("=" * 60)
    logger.info(f"  Seed={args.seed} | Tickers={cfg.data.tickers} | N_régimes={cfg.hmm.n_regimes}")
    logger.info(f"  Période : {cfg.data.start_date} → {cfg.data.end_date}")
    logger.info(f"  Mode allocation : {'soft (forward filtering)' if use_soft_alloc else 'hard (Viterbi)'}")

    # ------------------------------------------------------------------
    # 2. Données
    # ------------------------------------------------------------------
    logger.info("\n[1/5] 📊 Téléchargement & Feature Engineering...")
    from data.processor import MarketDataProcessor

    processor = MarketDataProcessor(cfg.data)
    bundle = processor.run(force_refresh=args.force_refresh)
    logger.success(
        f"  DataBundle — shape=({bundle.seq_len}, {bundle.n_features}) | "
        f"Train/Val/Test: {bundle.train_size}/{bundle.val_size}/{bundle.test_size}"
    )

    # ------------------------------------------------------------------
    # 3. VAE
    # ------------------------------------------------------------------
    logger.info("\n[2/5] 🧠 Entraînement du LSTM-VAE...")
    from models.trainer import VAETrainer

    cfg.vae.input_dim = bundle.n_features
    trainer = VAETrainer(cfg.vae, cfg.model_dir)
    train_history = None

    checkpoint_path = cfg.model_dir / "vae_best.pt"
    if args.skip_training and checkpoint_path.exists():
        logger.info("  → Chargement du checkpoint VAE existant")
        vae_model = trainer.load()
    else:
        vae_model, train_history = trainer.train(bundle)
        logger.success(
            f"  VAE entraîné — {len(train_history.val_loss)} epochs | "
            f"best_epoch={train_history.best_epoch}"
        )

    latent_train, latent_val, latent_test = trainer.encode_all(vae_model, bundle)
    logger.info(f"  Espace latent : dim={latent_train.shape[1]}")

    # ------------------------------------------------------------------
    # 4. HMM & Baseline
    # ------------------------------------------------------------------
    logger.info("\n[3/5] 🎯 Fitting HMM & Baseline Markov-Switching...")
    from models.hmm import RegimeHMM
    from models.markov_switching import MarkovSwitchingBaseline

    regime_hmm = RegimeHMM(cfg.hmm)

    # ---------------------------------------------------------------
    # [FIX #1] : passer returns_train au HMM pour un ordre de régimes
    # financièrement correct (tri par rendement réel, pas variance latente)
    #
    # Les latents sont des séquences glissantes de longueur seq_len.
    # latent_train[i] correspond à la séquence [i, i+seq_len].
    # Le rendement correspondant est celui à t = i + seq_len - 1.
    # On aligne donc les rendements sur les DERNIERS seq_len jours du train.
    # ---------------------------------------------------------------
    n_latents_train = latent_train.shape[0]
    returns_train_values = bundle.returns_train.values   # np.ndarray (N_train_full,)

    # Les N latents correspondent aux N derniers pas de temps
    # (les premiers seq_len-1 ont été consommés comme lookback)
    returns_train_aligned = returns_train_values[-n_latents_train:]

    if len(returns_train_aligned) != n_latents_train:
        logger.warning(
            f"  Alignement returns/latents : {len(returns_train_aligned)} vs "
            f"{n_latents_train} — fallback sur variance latente"
        )
        returns_train_aligned = None

    regime_hmm.fit(latent_train, returns_market=returns_train_aligned)
    if returns_train_aligned is not None:
        regime_hmm.validate_regime_quality(latent_train, returns_train_aligned)

    # ---------------------------------------------------------------
    # [FIX #2] : prédiction causale (forward filtering) pour le test set.
    #
    # On garde les deux versions :
    #   - regimes_test_causal : pour la stratégie (sans look-ahead)
    #   - regimes_test_viterbi : pour la visualisation uniquement
    #   - regime_proba_test   : probabilités causales forward (pour les plots)
    # ---------------------------------------------------------------
    logger.info("  Calcul des régimes (causal forward filtering)...")
    regimes_test_causal = regime_hmm.predict_causal(latent_test)   # sans look-ahead

    logger.info("  Calcul des régimes Viterbi (pour visualisation uniquement)...")
    regimes_test_viterbi = regime_hmm.predict(latent_test)         # avec look-ahead

    # Probabilités forward causales (pour plots et soft alloc)
    regime_proba_test = regime_hmm.predict_proba_causal(latent_test)  # (N, K) causal

    transition_matrix = regime_hmm.get_transition_matrix()
    regime_hmm.save(cfg.model_dir / "hmm.pkl")

    # Log de la différence Viterbi vs causal (indicateur de look-ahead bias)
    agreement_rate = (regimes_test_causal == regimes_test_viterbi).mean()
    logger.info(
        f"  Accord Viterbi/Causal : {agreement_rate:.1%} "
        f"({'faible look-ahead bias' if agreement_rate > 0.9 else 'look-ahead bias significatif détecté'})"
    )

    # Baseline Hamilton
    baseline = MarkovSwitchingBaseline(cfg.markov)
    try:
        baseline.fit(bundle.returns_train)
        regimes_baseline_test = baseline.predict(bundle.returns_test)
        baseline.print_summary()
        logger.success("  Baseline Hamilton ajusté.")
    except Exception as exc:
        logger.warning(f"  Hamilton indisponible ({exc}). Fallback binaire.")
        regimes_baseline_test = (regimes_test_causal > 0).astype(int)

    # ------------------------------------------------------------------
    # 5. Backtest & Évaluation
    # ------------------------------------------------------------------
    logger.info("\n[4/5] 📈 Backtest de la stratégie adaptative...")
    from strategy.backtester import AdaptiveStrategyBacktester

    backtester = AdaptiveStrategyBacktester(cfg.strategy)

    # Alignement temporel : les séquences glissantes (seq_len) réduisent
    # le nombre de prédictions. On aligne prices_test sur les régimes.
    n_preds = len(regimes_test_causal)
    prices_aligned = bundle.prices_test.iloc[-n_preds:]
    baseline_aligned = regimes_baseline_test[-n_preds:]

    # ---------------------------------------------------------------
    # [FIX #2 suite] : choix entre soft allocation et hard Viterbi
    #
    # Mode soft (recommandé, par défaut) :
    #   On calcule des régimes "filtrés par confiance" à partir des
    #   probabilités forward. Les jours où le modèle est incertain
    #   (confidence < min_confidence) reviennent au régime de transition.
    #
    # Mode hard :
    #   On utilise directement les régimes causaux (argmax du forward).
    #   Plus simple mais plus sensible au bruit de classification.
    # ---------------------------------------------------------------
    if use_soft_alloc:
        logger.info("  Mode soft allocation (filtre de confiance activé)...")
        regimes_for_backtest = compute_soft_regimes(
            regime_hmm=regime_hmm,
            latent_test=latent_test,
            regime_allocations=cfg.strategy.regime_allocations,
            min_confidence=cfg.strategy.min_confidence,
        )
    else:
        logger.info("  Mode hard allocation (Viterbi causal)...")
        regimes_for_backtest = regimes_test_causal

    backtest_results = backtester.run(
        prices=prices_aligned,
        regimes=regimes_for_backtest,
    )

    # Garde aussi les résultats avec Viterbi brut pour comparaison interne
    logger.info("  Backtest Viterbi brut (pour comparaison diagnostique)...")
    backtest_viterbi = backtester.run(
        prices=prices_aligned,
        regimes=regimes_test_viterbi,
    )
    _log_viterbi_vs_causal(backtest_results, backtest_viterbi)

    # ------------------------------------------------------------------
    # 6. Évaluation comparative & visualisations
    # ------------------------------------------------------------------
    logger.info("\n[5/5] 📊 Évaluation comparative & visualisations...")
    from evaluation.comparator import ModelComparator
    from utils.plotting import RegimePlotter

    comparator = ModelComparator(cfg)
    report = comparator.compare(
        prices=prices_aligned,
        regimes_vae_hmm=regimes_for_backtest,    # régimes effectivement utilisés
        regimes_baseline=baseline_aligned,
        backtest_results=backtest_results,
    )
    comparator.print_summary(report)
    comparator.save_report(report, cfg.results_dir / "comparison_report.json")

    # [NOUVEAU] Sauvegarde également les régimes causaux pour analyse
    _save_regime_diagnostics(
        regimes_causal=regimes_test_causal,
        regimes_viterbi=regimes_test_viterbi,
        regimes_effective=regimes_for_backtest,
        forward_proba=regime_proba_test,
        prices=prices_aligned,
        output_dir=cfg.results_dir,
    )

    plotter = RegimePlotter(cfg.figures_dir)
    plotter.plot_all(
        prices=prices_aligned,
        regimes_vae_hmm=regimes_for_backtest,    # régimes effectifs pour les plots
        regimes_baseline=baseline_aligned,
        backtest_results=backtest_results,
        train_history=train_history.to_dict() if train_history else None,
        regime_proba=regime_proba_test,          # probabilités causales
        latent_vectors=latent_test,
        transition_matrix=transition_matrix,
    )

    # ------------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("  RÉSULTATS FINAUX")
    logger.info("=" * 60)
    for model_name, metrics in report["metrics"].items():
        logger.info(f"\n  [{model_name}]")
        logger.info(f"    CAGR         : {metrics.get('annualized_return', 0):.2%}")
        logger.info(f"    Sharpe       : {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"    Sortino      : {metrics.get('sortino_ratio', 0):.3f}")
        logger.info(f"    Max Drawdown : {metrics.get('max_drawdown', 0):.2%}")

    ic = report.get("information_coefficient", {})
    logger.info(f"\n  IC régime (1j) : {ic.get('ic_1d', 0):.4f}")
    logger.info(f"  IC régime (5j) : {ic.get('ic_5d', 0):.4f}")
    logger.info(f"\n  ⏱  Pipeline terminé en {elapsed:.1f}s")
    logger.info(f"  📁 Figures     → {cfg.figures_dir}")
    logger.info(f"  📄 Rapport     → {cfg.results_dir / 'comparison_report.json'}")
    logger.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# Helpers diagnostiques (nouvelles fonctions)
# ---------------------------------------------------------------------------

def _log_viterbi_vs_causal(
    result_effective,
    result_viterbi,
) -> None:
    """
    Compare les métriques entre la stratégie effective et Viterbi brut.

    Permet de quantifier l'impact du look-ahead bias : si Viterbi brut
    est significativement meilleur, c'est un signe de biais.
    """
    try:
        from config.constants import MODEL_VAE_HMM
        sharpe_eff = result_effective[MODEL_VAE_HMM].metrics.get("sharpe_ratio", 0)
        sharpe_vit = result_viterbi[MODEL_VAE_HMM].metrics.get("sharpe_ratio", 0)
        diff = sharpe_vit - sharpe_eff

        if diff > 0.1:
            logger.warning(
                f"  ⚠ Look-ahead bias détecté : Viterbi brut Sharpe={sharpe_vit:.3f} "
                f"vs Causal Sharpe={sharpe_eff:.3f} (Δ={diff:+.3f}). "
                f"Utilise la stratégie causale pour le reporting."
            )
        else:
            logger.info(
                f"  ✓ Faible look-ahead bias : Viterbi={sharpe_vit:.3f}, "
                f"Causal={sharpe_eff:.3f} (Δ={diff:+.3f})"
            )
    except Exception:
        pass  # Non critique


def _save_regime_diagnostics(
    regimes_causal: np.ndarray,
    regimes_viterbi: np.ndarray,
    regimes_effective: np.ndarray,
    forward_proba: np.ndarray,
    prices: pd.Series,
    output_dir: Path,
) -> None:
    """
    Sauvegarde un CSV de diagnostic des régimes pour analyse post-hoc.

    Colonnes :
    - date             : date du jour
    - regime_causal    : régime forward filtering (sans look-ahead)
    - regime_viterbi   : régime Viterbi (avec look-ahead, biaisé)
    - regime_effective : régime utilisé pour le backtest
    - proba_0..K-1     : probabilités a posteriori causales
    - confidence       : max(proba) = confiance de la prédiction
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n = len(regimes_causal)
        df = pd.DataFrame(index=prices.index[-n:] if hasattr(prices, 'index') else range(n))
        df["regime_causal"] = regimes_causal
        df["regime_viterbi"] = regimes_viterbi
        df["regime_effective"] = regimes_effective

        for k in range(forward_proba.shape[1]):
            df[f"proba_{k}"] = forward_proba[:, k]

        df["confidence"] = forward_proba.max(axis=1)
        df["is_uncertain"] = df["confidence"] < 0.55

        output_path = output_dir / "regime_diagnostics.csv"
        df.to_csv(output_path)
        logger.info(f"  Diagnostics régimes → {output_path}")

    except Exception as exc:
        logger.warning(f"  Impossible de sauvegarder les diagnostics : {exc}")


if __name__ == "__main__":
    sys.exit(main())
