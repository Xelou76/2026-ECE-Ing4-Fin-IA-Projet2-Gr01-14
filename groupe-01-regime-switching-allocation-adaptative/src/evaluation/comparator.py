"""
evaluation/comparator.py
=========================
Comparaison scientifique des modèles : VAE-HMM vs Markov-Switching vs Buy-and-Hold.

Ce module est le cœur de l'évaluation quantitative du projet :
1. Compare les métriques de performance des 3 stratégies.
2. Calcule l'Information Coefficient des signaux de régime.
3. Produit un rapport JSON structuré et un tableau comparatif.
4. Implémente le test statistique de Jobson-Korkie (différence de Sharpe).

Classes
-------
ModelComparator
    Orchestre la comparaison complète des modèles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from config.constants import (
    METRICS_NAMES,
    MODEL_BUY_HOLD,
    MODEL_MARKOV_SWITCHING,
    MODEL_VAE_HMM,
    TRADING_DAYS_PER_YEAR,
)
from config.settings import ProjectConfig, get_settings
from strategy.backtester import AdaptiveStrategyBacktester, BacktestResult
from utils.metrics import compute_metrics, rolling_sharpe


class ModelComparator:
    """
    Compare les performances de VAE-HMM, Markov-Switching et Buy-and-Hold.

    Génère :
    - Un tableau de métriques comparatif (DataFrame)
    - Un rapport JSON structuré
    - Des statistiques de qualité des régimes (IC, durées, fréquences)
    - Un test de significativité de la différence de Sharpe (Jobson-Korkie)

    Parameters
    ----------
    cfg : ProjectConfig
        Configuration globale du projet.

    Examples
    --------
    >>> comparator = ModelComparator(cfg)
    >>> report = comparator.compare(
    ...     prices=prices_test,
    ...     regimes_vae_hmm=regimes_test,
    ...     regimes_baseline=regimes_baseline_test,
    ...     backtest_results=results,
    ... )
    >>> comparator.save_report(report, Path("results/report.json"))
    """

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self._backtester = AdaptiveStrategyBacktester(cfg.strategy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        prices: pd.Series,
        regimes_vae_hmm: np.ndarray,
        regimes_baseline: np.ndarray,
        backtest_results: Dict[str, BacktestResult],
        prices_bond: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Orchestre la comparaison complète des modèles.

        Calcule :
        1. Le backtest de la baseline Markov-Switching.
        2. Les métriques de qualité des régimes (IC, durées).
        3. Le test statistique de différence de Sharpe.
        4. Le tableau comparatif consolidé.

        Parameters
        ----------
        prices : pd.Series
            Prix de clôture du benchmark sur la période test.
        regimes_vae_hmm : np.ndarray
            Régimes prédits par VAE-HMM — shape (N,).
        regimes_baseline : np.ndarray
            Régimes prédits par Markov-Switching — shape (N,).
        backtest_results : Dict[str, BacktestResult]
            Résultats issus de ``AdaptiveStrategyBacktester.run()``.
        prices_bond : pd.Series, optional
            Prix de l'actif obligataire.

        Returns
        -------
        Dict[str, Any]
            Rapport complet : métriques, statistiques régimes, tests.
        """
        logger.info("ModelComparator — démarrage de la comparaison")

        # 1. Backtest de la baseline Markov-Switching
        baseline_result = self._backtester.run_baseline(
            prices=prices,
            regimes_baseline=regimes_baseline,
            prices_bond=prices_bond,
        )
        # Merge avec les résultats existants
        all_results = dict(backtest_results)
        all_results[MODEL_MARKOV_SWITCHING] = baseline_result

        # 2. Tableau comparatif des métriques
        metrics_table = self._build_metrics_table(all_results)
        logger.info(f"\n{metrics_table.to_string(float_format=lambda x: f'{x:.4f}')}\n")

        # 3. Statistiques de qualité des régimes
        regime_stats = self._compute_regime_stats(
            returns=prices.pct_change().dropna(),
            regimes_vae=regimes_vae_hmm,
            regimes_baseline=regimes_baseline,
        )

        # 4. Test statistique de Sharpe
        sharpe_test = self._jobson_korkie_test(
            returns_a=all_results[MODEL_VAE_HMM].returns,
            returns_b=all_results[MODEL_BUY_HOLD].returns,
            name_a=MODEL_VAE_HMM,
            name_b=MODEL_BUY_HOLD,
        )

        # 5. IC du signal de régime
        ic_1d = self._information_coefficient(
            regimes=regimes_vae_hmm,
            returns=prices.pct_change().dropna(),
            horizon=1,
        )
        ic_5d = self._information_coefficient(
            regimes=regimes_vae_hmm,
            returns=prices.pct_change().dropna(),
            horizon=5,
        )

        report: Dict[str, Any] = {
            "metrics": {
                name: res.metrics for name, res in all_results.items()
            },
            "metrics_table": metrics_table.to_dict(),
            "regime_stats": regime_stats,
            "sharpe_significance_test": sharpe_test,
            "information_coefficient": {"ic_1d": ic_1d, "ic_5d": ic_5d},
            "n_observations": len(prices),
            "period": {
                "start": str(prices.index[0].date()) if hasattr(prices.index[0], 'date') else str(prices.index[0]),
                "end": str(prices.index[-1].date()) if hasattr(prices.index[-1], 'date') else str(prices.index[-1]),
            },
        }

        logger.success("Comparaison terminée.")
        return report

    def save_report(self, report: Dict[str, Any], path: Path) -> None:
        """
        Sauvegarde le rapport de comparaison en JSON.

        Parameters
        ----------
        report : Dict[str, Any]
            Rapport issu de ``compare()``.
        path : Path
            Chemin de sortie (.json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sérialisation robuste (gère les types NumPy)
        def _serialize(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_serialize)

        logger.info(f"Rapport sauvegardé : {path}")

    def print_summary(self, report: Dict[str, Any]) -> None:
        """
        Affiche un résumé formaté du rapport dans les logs.

        Parameters
        ----------
        report : Dict[str, Any]
            Rapport issu de ``compare()``.
        """
        sep = "═" * 60
        logger.info(f"\n{sep}")
        logger.info("  COMPARAISON DES MODÈLES — RÉSUMÉ")
        logger.info(f"{sep}")

        for model_name, metrics in report["metrics"].items():
            logger.info(f"\n  ▶ {model_name}")
            logger.info(f"    CAGR              : {metrics.get('annualized_return', 0):.2%}")
            logger.info(f"    Sharpe Ratio      : {metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"    Sortino Ratio     : {metrics.get('sortino_ratio', 0):.3f}")
            logger.info(f"    Max Drawdown      : {metrics.get('max_drawdown', 0):.2%}")
            logger.info(f"    Calmar Ratio      : {metrics.get('calmar_ratio', 0):.3f}")

        ic = report.get("information_coefficient", {})
        logger.info(f"\n  IC régime (1j)    : {ic.get('ic_1d', 0):.4f}")
        logger.info(f"  IC régime (5j)    : {ic.get('ic_5d', 0):.4f}")

        st = report.get("sharpe_significance_test", {})
        if st:
            sig = "★ SIGNIFICATIF (p<0.05)" if st.get("significant_5pct") else "(non significatif)"
            logger.info(
                f"\n  Test Jobson-Korkie ({MODEL_VAE_HMM} vs {MODEL_BUY_HOLD}) : "
                f"ΔSharpe={st.get('delta_sharpe', 0):.3f} | "
                f"p={st.get('p_value', 1):.3f} {sig}"
            )
        logger.info(f"{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metrics_table(
        results: Dict[str, BacktestResult],
    ) -> pd.DataFrame:
        """Construit le DataFrame comparatif des métriques."""
        _DISPLAY = [
            ("annualized_return", "CAGR"),
            ("annualized_volatility", "Vol. ann."),
            ("sharpe_ratio", "Sharpe"),
            ("sortino_ratio", "Sortino"),
            ("calmar_ratio", "Calmar"),
            ("max_drawdown", "Max DD"),
            ("total_return", "Return total"),
            ("win_rate", "Win Rate"),
            ("n_trades", "Nb rebalancements"),
        ]
        rows: Dict[str, Dict[str, float]] = {}
        for key, label in _DISPLAY:
            rows[label] = {
                name: res.metrics.get(key, np.nan)
                for name, res in results.items()
            }
        return pd.DataFrame(rows).T

    @staticmethod
    def _compute_regime_stats(
        returns: pd.Series,
        regimes_vae: np.ndarray,
        regimes_baseline: np.ndarray,
    ) -> Dict[str, Any]:
        """Calcule les statistiques conditionnelles par régime."""
        from config.constants import REGIME_LABELS, BASELINE_REGIME_LABELS

        def _stats_per_regime(
            ret: pd.Series, reg: np.ndarray, labels: Dict[int, str]
        ) -> Dict[str, Any]:
            stats_dict: Dict[str, Any] = {}
            n_regimes = int(max(reg)) + 1
            for k in range(n_regimes):
                mask = np.array(reg) == k
                r_k = ret.values[mask[: len(ret)]] if len(mask) > len(ret) else ret.values[mask]
                label = labels.get(k, f"Regime {k}")
                durations = _compute_durations(reg, k)
                stats_dict[label] = {
                    "frequency_pct": float(mask.mean() * 100),
                    "mean_daily_return_pct": float(r_k.mean() * 100) if len(r_k) > 0 else 0.0,
                    "ann_volatility_pct": float(r_k.std() * np.sqrt(252) * 100) if len(r_k) > 1 else 0.0,
                    "mean_duration_days": float(np.mean(durations)) if durations else 0.0,
                    "max_duration_days": float(np.max(durations)) if durations else 0.0,
                    "n_episodes": len(durations),
                }
            return stats_dict

        def _compute_durations(reg: np.ndarray, target: int) -> List[int]:
            durs, count = [], 0
            for r in reg:
                if r == target:
                    count += 1
                elif count > 0:
                    durs.append(count)
                    count = 0
            if count > 0:
                durs.append(count)
            return durs

        from typing import List
        n = min(len(returns), len(regimes_vae))
        ret_aligned = returns.iloc[-n:] if len(returns) >= n else returns

        return {
            "vae_hmm": _stats_per_regime(ret_aligned, regimes_vae[:n], REGIME_LABELS),
            "markov_switching": _stats_per_regime(
                ret_aligned, regimes_baseline[:n], BASELINE_REGIME_LABELS
            ),
        }

    @staticmethod
    def _jobson_korkie_test(
        returns_a: pd.Series,
        returns_b: pd.Series,
        name_a: str = "A",
        name_b: str = "B",
    ) -> Dict[str, float]:
        """
        Test de Jobson-Korkie (1981) : H0 : SR(A) = SR(B).

        Teste si la différence de Sharpe entre deux stratégies est
        statistiquement significative.

        Parameters
        ----------
        returns_a, returns_b : pd.Series
            Returns journaliers des deux stratégies.
        name_a, name_b : str
            Noms pour les logs.

        Returns
        -------
        Dict[str, float]
            z-stat, p-value, delta_sharpe, significant_5pct.
        """
        n = min(len(returns_a), len(returns_b))
        ra = returns_a.dropna().values[:n]
        rb = returns_b.dropna().values[:n]
        n = len(ra)

        mu_a, mu_b = ra.mean(), rb.mean()
        s_a, s_b = ra.std(), rb.std()
        s_ab = np.cov(ra, rb)[0, 1]

        sr_a = mu_a / s_a * np.sqrt(252) if s_a > 0 else 0.0
        sr_b = mu_b / s_b * np.sqrt(252) if s_b > 0 else 0.0
        delta = sr_a - sr_b

        # Variance asymptotique (Jobson-Korkie, 1981 — formule complète)
        var_num = (
            2 * s_a**2 * s_b**2
            - 2 * s_a * s_b * s_ab
            + 0.5 * mu_a**2 * s_b**2
            + 0.5 * mu_b**2 * s_a**2
            - (mu_a * mu_b * s_ab**2) / (s_a * s_b)
        )
        var_denom = s_a**2 * s_b**2
        var_sr_diff = var_num / (n * var_denom) if var_denom > 1e-10 else 0.0

        z_stat = delta / np.sqrt(max(var_sr_diff, 1e-10))
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        result = {
            "sharpe_a": round(float(sr_a), 4),
            "sharpe_b": round(float(sr_b), 4),
            "delta_sharpe": round(float(delta), 4),
            "z_statistic": round(float(z_stat), 4),
            "p_value": round(p_value, 4),
            "significant_5pct": bool(p_value < 0.05),
        }

        logger.info(
            f"  Jobson-Korkie ({name_a} vs {name_b}) : "
            f"ΔSharpe={delta:.3f} | z={z_stat:.2f} | p={p_value:.3f} "
            f"{'★ significatif' if p_value < 0.05 else '(non sig.)'}"
        )
        return result

    @staticmethod
    def _information_coefficient(
        regimes: np.ndarray,
        returns: pd.Series,
        horizon: int = 1,
    ) -> float:
        """
        Calcule l'IC (Information Coefficient) du signal de régime.

        IC = corrélation de Spearman entre -régime_t et return_{t+horizon}.
        Un IC positif signifie que le régime prédit correctement la direction.

        Parameters
        ----------
        regimes : np.ndarray
            Régimes prédits.
        returns : pd.Series
            Returns du benchmark.
        horizon : int
            Horizon de prédiction (en jours).

        Returns
        -------
        float
            Information Coefficient (corrélation Spearman).
        """
        n = min(len(regimes), len(returns))
        signal = -regimes[:n - horizon].astype(float)
        fwd_returns = returns.values[horizon:n]

        if len(signal) < 10:
            return 0.0

        ic, p_val = stats.spearmanr(signal, fwd_returns)
        logger.debug(f"  IC(horizon={horizon}j) = {ic:.4f} (p={p_val:.3f})")
        return float(ic)
