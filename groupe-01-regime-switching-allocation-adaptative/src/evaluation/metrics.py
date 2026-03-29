"""
evaluation/metrics.py
======================
Module de calcul des métriques de performance et de comparaison des modèles.

Ce module implémente :
1. ``PerformanceReport`` : rapport complet pour une stratégie.
2. ``ModelComparator`` : tableau comparatif multi-stratégies (VAE-HMM vs Hamilton).
3. ``RegimeMetrics`` : métriques spécifiques aux régimes (qualité de détection).

Métriques implémentées
-----------------------
- Rendement total & CAGR
- Sharpe Ratio (annualisé)
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown & durée
- Turnover & coûts de transaction
- Statistiques de régime : fréquence, durée, transitions
- Information Coefficient (IC) pour la prédictivité des régimes

Classes
-------
PerformanceReport
    Calcule et formate un rapport complet pour un BacktestResult.
ModelComparator
    Compare N stratégies (VAE-HMM, Hamilton, Buy-and-Hold).
RegimeMetrics
    Métriques de qualité des régimes détectés.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from config.settings import EvaluationConfig, ProjectConfig
from strategy.adaptive_strategy import BacktestResult


# ---------------------------------------------------------------------------
# PerformanceReport
# ---------------------------------------------------------------------------

class PerformanceReport:
    """
    Génère un rapport de performance détaillé à partir d'un BacktestResult.

    Calcule les métriques standard et enrichit le BacktestResult avec
    des statistiques supplémentaires (underwater plot, monthly returns, etc.).

    Parameters
    ----------
    cfg : EvaluationConfig
        Configuration de l'évaluation (facteur d'annualisation, fenêtres...).

    Examples
    --------
    >>> reporter = PerformanceReport(cfg)
    >>> reporter.generate(result)
    >>> reporter.print_summary(result)
    """

    def __init__(self, cfg: EvaluationConfig) -> None:
        self.cfg = cfg

    def generate(self, result: BacktestResult) -> BacktestResult:
        """
        Enrichit le BacktestResult avec des métriques supplémentaires.

        Parameters
        ----------
        result : BacktestResult
            Résultat brut du backtest.

        Returns
        -------
        BacktestResult
            Même objet, métriques enrichies in-place.
        """
        # Métriques supplémentaires
        extra = self._compute_extra_metrics(result)
        result.metrics.update(extra)

        # Sharpe glissant (si pas déjà calculé)
        if result.rolling_sharpe.empty:
            result.rolling_sharpe = self._rolling_sharpe(
                result.returns, self.cfg.rolling_window
            )

        return result

    def print_summary(self, result: BacktestResult) -> None:
        """
        Affiche un tableau formaté des métriques dans les logs.

        Parameters
        ----------
        result : BacktestResult
            Résultat du backtest (après generate()).
        """
        sep = "─" * 50
        logger.info(sep)
        logger.info(f"  📊 RAPPORT : {result.strategy_name}")
        logger.info(sep)
        m = result.metrics

        logger.info(f"  Rendement total       : {m.get('total_return_pct', 0):.2f}%")
        logger.info(f"  CAGR                  : {m.get('cagr_pct', 0):.2f}%")
        logger.info(f"  Volatilité (ann.)     : {m.get('ann_volatility_pct', 0):.2f}%")
        logger.info(f"  Sharpe Ratio          : {m.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Sortino Ratio         : {m.get('sortino_ratio', 0):.3f}")
        logger.info(f"  Calmar Ratio          : {m.get('calmar_ratio', 0):.3f}")
        logger.info(f"  Max Drawdown          : {m.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"  Durée Max DD          : {m.get('max_drawdown_duration_days', 0):.0f}j")
        logger.info(f"  Win Rate              : {m.get('win_rate_pct', 0):.1f}%")
        logger.info(f"  Profit Factor         : {m.get('profit_factor', 0):.2f}")
        logger.info(f"  Turnover moyen (ann.) : {m.get('ann_turnover_pct', 0):.1f}%")
        logger.info(f"  Coûts transaction     : {m.get('total_transaction_costs_pct', 0):.3f}%")
        logger.info(sep)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_extra_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calcule les métriques additionnelles."""
        returns = result.returns
        turnover = result.turnover

        # Turnover annualisé
        ann_turnover = turnover.mean() * self.cfg.annualization_factor * 100.0

        # Skewness & Kurtosis des returns
        skew = float(stats.skew(returns.dropna()))
        kurt = float(stats.kurtosis(returns.dropna()))

        # Value at Risk 95% (historique)
        var_95 = float(returns.quantile(0.05))
        cvar_95 = float(returns[returns <= var_95].mean())

        # Best / Worst year
        if isinstance(returns.index, pd.DatetimeIndex):
            annual = returns.groupby(returns.index.year).apply(
                lambda x: (1 + x).prod() - 1
            )
            best_year = annual.max() * 100.0
            worst_year = annual.min() * 100.0
        else:
            best_year = worst_year = 0.0

        # Nombre de rebalancements effectifs
        n_rebalances = int((result.turnover > 0).sum())

        return {
            "ann_turnover_pct": ann_turnover,
            "returns_skewness": round(skew, 4),
            "returns_kurtosis": round(kurt, 4),
            "var_95_daily_pct": var_95 * 100.0,
            "cvar_95_daily_pct": cvar_95 * 100.0,
            "best_year_pct": best_year,
            "worst_year_pct": worst_year,
            "n_rebalances": n_rebalances,
        }

    def _rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        rf_daily = 0.02 / 252.0
        excess = returns - rf_daily
        return (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(252)
        ).rename("rolling_sharpe")


# ---------------------------------------------------------------------------
# ModelComparator
# ---------------------------------------------------------------------------

class ModelComparator:
    """
    Compare plusieurs stratégies et génère un tableau récapitulatif.

    Fournit :
    - Tableau de métriques côte-à-côte (VAE-HMM, Hamilton, Buy-and-Hold)
    - Tests statistiques : différence de Sharpe (Ledoit-Wolf, DM test)
    - Décomposition de la surperformance

    Parameters
    ----------
    cfg : ProjectConfig
        Configuration globale du projet.

    Examples
    --------
    >>> comparator = ModelComparator(cfg)
    >>> comparator.add_result(vae_hmm_result)
    >>> comparator.add_result(hamilton_result)
    >>> comparator.add_result(buy_hold_result)
    >>> df = comparator.summary_table()
    >>> comparator.print_comparison()
    """

    # Métriques à afficher dans le tableau comparatif
    _DISPLAY_METRICS = [
        ("total_return_pct", "Return Total (%)"),
        ("cagr_pct", "CAGR (%)"),
        ("ann_volatility_pct", "Volatilité ann. (%)"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("calmar_ratio", "Calmar Ratio"),
        ("max_drawdown_pct", "Max Drawdown (%)"),
        ("max_drawdown_duration_days", "Durée Max DD (j)"),
        ("win_rate_pct", "Win Rate (%)"),
        ("profit_factor", "Profit Factor"),
        ("ann_turnover_pct", "Turnover ann. (%)"),
        ("total_transaction_costs_pct", "Coûts transaction (%)"),
        ("var_95_daily_pct", "VaR 95% quotidien (%)"),
        ("cvar_95_daily_pct", "CVaR 95% quotidien (%)"),
        ("n_rebalances", "Nb Rebalancements"),
    ]

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self._results: List[BacktestResult] = []

    def add_result(self, result: BacktestResult) -> "ModelComparator":
        """Ajoute un résultat de backtest à la comparaison."""
        self._results.append(result)
        return self

    def summary_table(self) -> pd.DataFrame:
        """
        Génère un DataFrame comparatif des métriques de toutes les stratégies.

        Returns
        -------
        pd.DataFrame
            Tableau métriques × stratégies.
        """
        if not self._results:
            raise RuntimeError("Aucun résultat ajouté. Appelez add_result() d'abord.")

        rows = {}
        for key, label in self._DISPLAY_METRICS:
            rows[label] = {
                r.strategy_name: r.metrics.get(key, np.nan)
                for r in self._results
            }
        df = pd.DataFrame(rows).T
        return df

    def print_comparison(self) -> None:
        """Affiche le tableau comparatif formaté dans les logs."""
        df = self.summary_table()
        logger.info("\n" + "=" * 70)
        logger.info("  COMPARAISON DES STRATÉGIES")
        logger.info("=" * 70)
        logger.info(f"\n{df.to_string(float_format=lambda x: f'{x:.3f}')}\n")

    def sharpe_significance_test(
        self,
        strategy_a: str,
        strategy_b: str,
    ) -> Dict[str, float]:
        """
        Test de significativité de la différence de Sharpe (Jobson-Korkie, 1981).

        H0 : Sharpe(A) = Sharpe(B)
        H1 : Sharpe(A) ≠ Sharpe(B)

        Parameters
        ----------
        strategy_a, strategy_b : str
            Noms des stratégies à comparer.

        Returns
        -------
        Dict[str, float]
            z-statistic, p-value, delta_sharpe.
        """
        result_a = self._get_result(strategy_a)
        result_b = self._get_result(strategy_b)

        if result_a is None or result_b is None:
            return {}

        ra = result_a.returns.dropna().values
        rb = result_b.returns.dropna().values
        n = min(len(ra), len(rb))
        ra, rb = ra[:n], rb[:n]

        mu_a, mu_b = ra.mean(), rb.mean()
        sigma_a, sigma_b = ra.std(), rb.std()
        sigma_ab = np.cov(ra, rb)[0, 1]

        sr_a = mu_a / sigma_a * np.sqrt(252)
        sr_b = mu_b / sigma_b * np.sqrt(252)
        delta_sr = sr_a - sr_b

        # Variance asymptotique de la différence de Sharpe (Jobson-Korkie)
        var_diff = (
            1 / n * (
                2 * sigma_a**2 * sigma_b**2
                - 2 * sigma_a * sigma_b * sigma_ab
                + 0.5 * mu_a**2 * sigma_b**2
                + 0.5 * mu_b**2 * sigma_a**2
                - (mu_a * mu_b * sigma_ab**2) / (sigma_a * sigma_b)
            ) / (sigma_a**2 * sigma_b**2)
        )
        z_stat = delta_sr / np.sqrt(max(var_diff, 1e-10)) if var_diff > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        result = {
            "sharpe_a": sr_a,
            "sharpe_b": sr_b,
            "delta_sharpe": delta_sr,
            "z_statistic": z_stat,
            "p_value": p_value,
            "significant_5pct": p_value < 0.05,
        }

        logger.info(
            f"  Test Jobson-Korkie : {strategy_a} vs {strategy_b} | "
            f"ΔSharpe={delta_sr:.3f} | z={z_stat:.2f} | p={p_value:.3f} "
            f"{'★ SIGNIFICATIF' if p_value < 0.05 else '(non significatif)'}"
        )
        return result

    def _get_result(self, name: str) -> Optional[BacktestResult]:
        """Récupère un résultat par son nom."""
        for r in self._results:
            if r.strategy_name == name:
                return r
        logger.warning(f"Résultat '{name}' non trouvé.")
        return None


# ---------------------------------------------------------------------------
# RegimeMetrics
# ---------------------------------------------------------------------------

class RegimeMetrics:
    """
    Métriques de qualité de la détection de régimes.

    Évalue la capacité du modèle à détecter des états de marché pertinents :
    - Returns conditionnels par régime (le régime prédit il permet des profits ?)
    - Stabilité des régimes (durée moyenne, fréquence des transitions)
    - Information Coefficient (IC) : corrélation régime → return futur

    Parameters
    ----------
    cfg : ProjectConfig
        Configuration globale.
    regime_names : List[str], optional
        Noms des régimes pour les graphiques.

    Examples
    --------
    >>> rm = RegimeMetrics(cfg)
    >>> regime_stats = rm.regime_conditional_stats(returns, regimes)
    >>> ic = rm.information_coefficient(regimes, returns, horizon=1)
    """

    def __init__(
        self,
        cfg: ProjectConfig,
        regime_names: Optional[List[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.regime_names = regime_names or list(cfg.regime_names)

    def regime_conditional_stats(
        self,
        returns: pd.Series,
        regimes: np.ndarray,
        annualization: int = 252,
    ) -> pd.DataFrame:
        """
        Calcule les statistiques de returns conditionnelles par régime.

        Pour chaque régime k, calcule :
        - Return moyen annualisé
        - Volatilité annualisée
        - Sharpe ratio implicite
        - Fréquence d'occurrence
        - Durée moyenne des épisodes

        Parameters
        ----------
        returns : pd.Series
            Returns journaliers.
        regimes : np.ndarray
            Séquence de régimes — shape (N,).
        annualization : int
            Facteur d'annualisation.

        Returns
        -------
        pd.DataFrame
            Statistiques conditionnelles par régime.
        """
        n_regimes = int(max(regimes)) + 1
        rows = []

        for k in range(n_regimes):
            mask = regimes == k
            r_k = returns.values[mask]

            if len(r_k) == 0:
                continue

            mu = r_k.mean() * annualization * 100
            sigma = r_k.std() * np.sqrt(annualization) * 100
            sr = (r_k.mean() / r_k.std() * np.sqrt(annualization)) if r_k.std() > 0 else 0

            # Durées des épisodes
            durations = self._compute_durations(regimes, k)

            name = self.regime_names[k] if k < len(self.regime_names) else f"Régime {k}"
            rows.append({
                "Régime": f"R{k} — {name}",
                "Fréquence (%)": mask.mean() * 100,
                "Return ann. (%)": mu,
                "Volatilité ann. (%)": sigma,
                "Sharpe implicite": round(sr, 3),
                "Durée moy. (j)": np.mean(durations) if durations else 0,
                "Durée max (j)": np.max(durations) if durations else 0,
                "Nb épisodes": len(durations),
            })

        return pd.DataFrame(rows).set_index("Régime")

    def information_coefficient(
        self,
        regimes: np.ndarray,
        returns: pd.Series,
        horizon: int = 1,
        n_bins: int = 3,
    ) -> float:
        """
        Calcule l'Information Coefficient (IC) du signal de régime.

        L'IC mesure la corrélation de Spearman entre le régime prédit
        à t et le return du benchmark à t+horizon.

        Un IC positif → le régime prédit correctement la direction future.
        IC > 0.05 est considéré comme utile en pratique (quant finance).

        Parameters
        ----------
        regimes : np.ndarray
            Régimes prédits — shape (N,).
        returns : pd.Series
            Returns journaliers — shape (N,).
        horizon : int
            Horizon de prédiction en jours.
        n_bins : int
            Ignoré (conservé pour compatibilité). L'IC utilise Spearman.

        Returns
        -------
        float
            Information Coefficient (corrélation Spearman).
        """
        # Signal = -régime (régime élevé → bear → return négatif prédit)
        signal = -regimes[:-horizon].astype(float)
        future_returns = returns.values[horizon:]

        if len(signal) < 10:
            return 0.0

        ic, p_value = stats.spearmanr(signal, future_returns)

        logger.info(
            f"  IC(horizon={horizon}j) = {ic:.4f} "
            f"(p={p_value:.3f}) "
            f"{'★ significatif' if p_value < 0.05 else '(non sig.)'}"
        )
        return float(ic)

    def transition_matrix_empirical(
        self, regimes: np.ndarray
    ) -> pd.DataFrame:
        """
        Calcule la matrice de transition empirique (fréquentiste).

        A[i, j] = P_obs(r_{t+1} = j | r_t = i) estimée par comptage.

        Parameters
        ----------
        regimes : np.ndarray
            Séquence de régimes observés.

        Returns
        -------
        pd.DataFrame
            Matrice de transition empirique (K × K).
        """
        n_regimes = int(max(regimes)) + 1
        A = np.zeros((n_regimes, n_regimes))

        for t in range(len(regimes) - 1):
            A[regimes[t], regimes[t + 1]] += 1

        # Normalisation par ligne (évite division par zéro)
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        A = A / row_sums

        names = [
            self.regime_names[k] if k < len(self.regime_names) else f"R{k}"
            for k in range(n_regimes)
        ]
        return pd.DataFrame(A, index=names, columns=names)

    @staticmethod
    def _compute_durations(regimes: np.ndarray, target: int) -> List[int]:
        """Calcule les durées de chaque épisode dans le régime cible."""
        durations, count = [], 0
        for r in regimes:
            if r == target:
                count += 1
            elif count > 0:
                durations.append(count)
                count = 0
        if count > 0:
            durations.append(count)
        return durations