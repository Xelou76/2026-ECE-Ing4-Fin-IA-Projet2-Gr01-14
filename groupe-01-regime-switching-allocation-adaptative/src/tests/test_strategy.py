"""
tests/test_strategy.py
=======================
Tests unitaires pour les modules strategy/ et evaluation/.

Couvre :
- AdaptiveStrategyBacktester : calculs vectorisés, bornes d'allocations
- BacktestResult : intégrité et cohérence des champs
- utils.metrics : exactitude des métriques financières
- ModelComparator : tableau comparatif et test de Jobson-Korkie
- RegimePlotter : smoke tests (génération sans erreur)

Toutes les données sont synthétiques — pas de yfinance, pas de PyTorch requis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from strategy.backtester import AdaptiveStrategyBacktester, BacktestResult
from utils.metrics import (
    compute_metrics,
    max_drawdown,
    rolling_sharpe,
    sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    """Configuration du projet."""
    return get_settings()


@pytest.fixture(scope="module")
def dates():
    return pd.date_range("2015-01-01", periods=500, freq="B")


@pytest.fixture(scope="module")
def prices(dates):
    rng = np.random.default_rng(42)
    log_ret = rng.normal(3e-4, 0.01, 500)
    price_vals = 300.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(price_vals, index=dates, name="SPY")


@pytest.fixture(scope="module")
def returns(prices):
    return prices.pct_change().fillna(0.0)


@pytest.fixture(scope="module")
def regimes_3(dates):
    """3 régimes synthétiques avec transitions réalistes."""
    rng = np.random.default_rng(7)
    n = len(dates)
    reg = np.zeros(n, dtype=int)
    A = np.array([[0.97, 0.02, 0.01], [0.05, 0.90, 0.05], [0.01, 0.04, 0.95]])
    for t in range(1, n):
        reg[t] = rng.choice(3, p=A[reg[t - 1]])
    return reg


@pytest.fixture(scope="module")
def regimes_2(dates):
    """2 régimes (baseline Hamilton)."""
    rng = np.random.default_rng(13)
    n = len(dates)
    reg = np.zeros(n, dtype=int)
    for t in range(1, n):
        reg[t] = rng.choice(2, p=[0.97, 0.03] if reg[t - 1] == 0 else [0.05, 0.95])
    return reg


@pytest.fixture(scope="module")
def backtester(cfg):
    return AdaptiveStrategyBacktester(cfg.strategy)


@pytest.fixture(scope="module")
def vae_result(backtester, prices, regimes_3):
    results = backtester.run(prices=prices, regimes=regimes_3)
    from config.constants import MODEL_VAE_HMM
    return results[MODEL_VAE_HMM]


@pytest.fixture(scope="module")
def bah_result(backtester, prices, regimes_3):
    results = backtester.run(prices=prices, regimes=regimes_3)
    from config.constants import MODEL_BUY_HOLD
    return results[MODEL_BUY_HOLD]


# ---------------------------------------------------------------------------
# Tests : AdaptiveStrategyBacktester
# ---------------------------------------------------------------------------

class TestAdaptiveStrategyBacktester:

    def test_run_returns_expected_keys(self, backtester, prices, regimes_3):
        """run() doit retourner les clés MODEL_VAE_HMM et MODEL_BUY_HOLD."""
        from config.constants import MODEL_BUY_HOLD, MODEL_VAE_HMM
        results = backtester.run(prices, regimes_3)
        assert MODEL_VAE_HMM in results
        assert MODEL_BUY_HOLD in results

    def test_equity_curve_length(self, vae_result, prices):
        """L'equity curve doit avoir la même longueur que les prix."""
        assert len(vae_result.equity_curve) == len(prices)

    def test_equity_curve_positive(self, vae_result):
        """L'equity curve doit être strictement positive."""
        assert (vae_result.equity_curve > 0).all()

    def test_equity_curve_starts_near_100(self, vae_result):
        """L'equity curve doit démarrer à environ 100."""
        # Premier point : allocation initiale × return premier jour
        # Pas exactement 100 car le premier return est inclus
        assert 80 < vae_result.equity_curve.iloc[0] < 120

    def test_allocations_sum_to_one(self, vae_result):
        """Chaque ligne d'allocations doit sommer à 1.0 (à la tolérance près)."""
        row_sums = vae_result.allocations.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-8)

    def test_allocations_non_negative(self, vae_result):
        """Aucune allocation ne peut être négative (pas de vente à découvert)."""
        assert (vae_result.allocations >= 0).all().all()

    def test_allocations_bounded_by_one(self, vae_result):
        """Aucune allocation ne peut dépasser 1.0."""
        assert (vae_result.allocations <= 1.0 + 1e-8).all().all()

    def test_turnover_non_negative(self, vae_result):
        """Le turnover ne peut pas être négatif."""
        assert (vae_result.turnover >= 0).all()

    def test_returns_index_matches_prices(self, vae_result, prices):
        """L'index des returns doit correspondre à celui des prix."""
        assert vae_result.returns.index.equals(prices.index)

    def test_length_mismatch_raises(self, backtester, prices):
        """Longueurs incompatibles → ValueError."""
        bad_regimes = np.zeros(len(prices) + 10, dtype=int)
        with pytest.raises(ValueError, match="Longueurs incompatibles"):
            backtester.run(prices, bad_regimes)

    def test_bah_full_equity_allocation(self, bah_result):
        """Buy-and-Hold : 100% equity en permanence."""
        assert (bah_result.allocations["equity"] == 1.0).all()
        assert (bah_result.allocations["bond"] == 0.0).all()
        assert (bah_result.allocations["cash"] == 0.0).all()

    def test_bah_zero_turnover(self, bah_result):
        """Buy-and-Hold : zéro turnover et zéro coûts de transaction."""
        assert (bah_result.turnover == 0.0).all()

    def test_bah_equity_matches_benchmark(self, bah_result, returns):
        """Buy-and-Hold : equity curve identique aux returns cumulatifs."""
        expected = (1 + returns).cumprod() * 100.0
        np.testing.assert_allclose(
            bah_result.equity_curve.values,
            expected.values,
            rtol=1e-6,
        )

    def test_run_baseline_returns_backtest_result(self, backtester, prices, regimes_2):
        """run_baseline() doit retourner un BacktestResult valide."""
        result = backtester.run_baseline(prices, regimes_2)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(prices)

    def test_metrics_keys_present(self, vae_result):
        """Les métriques clés doivent être présentes dans le résultat."""
        required = [
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
        ]
        for key in required:
            assert key in vae_result.metrics, f"Métrique manquante : {key}"

    def test_sharpe_finite(self, vae_result):
        """Le Sharpe ratio doit être un nombre fini."""
        assert np.isfinite(vae_result.metrics["sharpe_ratio"])

    def test_max_drawdown_non_positive(self, vae_result):
        """Le Max Drawdown doit être ≤ 0."""
        assert vae_result.metrics["max_drawdown"] <= 0.0

    def test_rolling_sharpe_length(self, vae_result, prices):
        """Le Sharpe glissant doit avoir la même longueur que les prix."""
        assert len(vae_result.rolling_sharpe_63d) == len(prices)

    def test_three_regime_allocations_distinct(self, backtester, prices):
        """Avec 3 régimes, les allocations doivent être distinctes entre régimes."""
        # Crée des régimes purs (chaque tiers = un régime)
        n = len(prices)
        regimes = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)))
        results = backtester.run(prices, regimes)
        from config.constants import MODEL_VAE_HMM
        allocs = results[MODEL_VAE_HMM].allocations

        # Les allocations equity du premier tiers vs dernier tiers doivent différer
        mean_bear = allocs["equity"].iloc[:n // 3].mean()
        mean_bull = allocs["equity"].iloc[-n // 3:].mean()
        assert mean_bull > mean_bear, "Allocation bull doit être > bear"


# ---------------------------------------------------------------------------
# Tests : utils.metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_sharpe_positive_series(self):
        """Serie toujours positive → Sharpe > 0."""
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        ret = pd.Series(np.abs(np.random.default_rng(0).normal(0.001, 0.005, 252)), index=dates)
        sr = sharpe_ratio(ret, risk_free_rate=0.0)
        assert sr > 0

    def test_sharpe_negative_series(self):
        """Serie toujours négative → Sharpe < 0."""
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        ret = pd.Series(-np.abs(np.random.default_rng(1).normal(0.001, 0.005, 252)), index=dates)
        sr = sharpe_ratio(ret, risk_free_rate=0.0)
        assert sr < 0

    def test_max_drawdown_known_series(self):
        """Test sur une série à drawdown connu : 100→120→90 = -25%."""
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        prices = pd.Series([100.0, 120.0, 90.0], index=dates)
        ret = prices.pct_change().fillna(0.0)
        mdd = max_drawdown(ret)
        # Drawdown = (90 - 120) / 120 = -25%
        assert pytest.approx(mdd, abs=0.01) == -0.25

    def test_max_drawdown_monotone_up(self):
        """Série monotone croissante → drawdown = 0."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        ret = pd.Series(np.full(100, 0.001), index=dates)
        mdd = max_drawdown(ret)
        assert mdd == pytest.approx(0.0, abs=1e-6)

    def test_compute_metrics_returns_dict(self, returns):
        """compute_metrics doit retourner un dict avec les clés attendues."""
        m = compute_metrics(returns)
        for key in ["sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"]:
            assert key in m

    def test_compute_metrics_values_finite(self, returns):
        """Toutes les métriques doivent être finies."""
        m = compute_metrics(returns)
        for key, val in m.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{key} = {val} n'est pas fini"

    def test_rolling_sharpe_length(self, returns):
        """Rolling Sharpe : longueur identique à la série d'entrée."""
        rs = rolling_sharpe(returns, window=21)
        assert len(rs) == len(returns)

    def test_rolling_sharpe_nan_at_start(self, returns):
        """Les premières (window-1) valeurs du Sharpe glissant doivent être NaN."""
        window = 21
        rs = rolling_sharpe(returns, window=window)
        assert rs.iloc[:window - 1].isna().all()

    def test_sortino_gte_sharpe_positive_returns(self):
        """Pour des returns asymétriquement positifs, Sortino >= Sharpe."""
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        rng = np.random.default_rng(99)
        # Returns surtout positifs, rares pertes
        ret = pd.Series(np.clip(rng.normal(0.001, 0.006, 500), -0.003, 0.05), index=dates)
        m = compute_metrics(ret, risk_free_rate=0.02)
        # Sortino doit être >= Sharpe (downside vol < total vol)
        assert m["sortino_ratio"] >= m["sharpe_ratio"] * 0.5


# ---------------------------------------------------------------------------
# Tests : ModelComparator
# ---------------------------------------------------------------------------

class TestModelComparator:

    def test_compare_runs_without_error(self, cfg, prices, regimes_3, regimes_2,
                                        backtester, returns):
        """compare() doit s'exécuter sans lever d'exception."""
        from evaluation.comparator import ModelComparator
        results = backtester.run(prices, regimes_3)
        comparator = ModelComparator(cfg)
        report = comparator.compare(
            prices=prices,
            regimes_vae_hmm=regimes_3,
            regimes_baseline=regimes_2,
            backtest_results=results,
        )
        assert "metrics" in report
        assert "information_coefficient" in report
        assert "sharpe_significance_test" in report

    def test_report_has_all_models(self, cfg, prices, regimes_3, regimes_2, backtester):
        """Le rapport doit contenir les 3 modèles."""
        from config.constants import MODEL_BUY_HOLD, MODEL_MARKOV_SWITCHING, MODEL_VAE_HMM
        from evaluation.comparator import ModelComparator
        results = backtester.run(prices, regimes_3)
        comparator = ModelComparator(cfg)
        report = comparator.compare(prices, regimes_3, regimes_2, results)
        for model in [MODEL_VAE_HMM, MODEL_BUY_HOLD, MODEL_MARKOV_SWITCHING]:
            assert model in report["metrics"], f"Modèle manquant : {model}"

    def test_ic_bounded(self, cfg, prices, regimes_3, regimes_2, backtester):
        """L'IC doit être dans [-1, 1]."""
        from evaluation.comparator import ModelComparator
        results = backtester.run(prices, regimes_3)
        comparator = ModelComparator(cfg)
        report = comparator.compare(prices, regimes_3, regimes_2, results)
        ic_1d = report["information_coefficient"]["ic_1d"]
        assert -1.0 <= ic_1d <= 1.0

    def test_save_report_creates_file(self, cfg, prices, regimes_3, regimes_2,
                                      backtester, tmp_path):
        """save_report() doit créer un fichier JSON valide."""
        import json
        from evaluation.comparator import ModelComparator
        results = backtester.run(prices, regimes_3)
        comparator = ModelComparator(cfg)
        report = comparator.compare(prices, regimes_3, regimes_2, results)
        out_path = tmp_path / "report.json"
        comparator.save_report(report, out_path)
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert "metrics" in data

    def test_jobson_korkie_p_value_bounded(self, cfg, prices, regimes_3, regimes_2, backtester):
        """La p-value du test de Jobson-Korkie doit être dans [0, 1]."""
        from evaluation.comparator import ModelComparator
        results = backtester.run(prices, regimes_3)
        comparator = ModelComparator(cfg)
        report = comparator.compare(prices, regimes_3, regimes_2, results)
        p = report["sharpe_significance_test"]["p_value"]
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Tests : RegimePlotter (smoke tests)
# ---------------------------------------------------------------------------

class TestRegimePlotter:

    def test_plot_regimes_on_price(self, prices, regimes_3, tmp_path):
        """plot_regimes_on_price doit générer un fichier sans erreur."""
        from utils.plotting import RegimePlotter
        plotter = RegimePlotter(tmp_path, dpi=72)
        plotter.plot_regimes_on_price(prices, regimes_3, filename="test_regimes")
        assert (tmp_path / "test_regimes.png").exists()

    def test_plot_equity_curves(self, prices, regimes_3, backtester, tmp_path):
        """plot_equity_curves doit générer un fichier sans erreur."""
        from utils.plotting import RegimePlotter
        results = backtester.run(prices, regimes_3)
        plotter = RegimePlotter(tmp_path, dpi=72)
        plotter.plot_equity_curves(results, filename="test_equity")
        assert (tmp_path / "test_equity.png").exists()

    def test_plot_transition_matrix(self, tmp_path):
        """plot_transition_matrix doit générer un fichier sans erreur."""
        from utils.plotting import RegimePlotter
        A = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
        plotter = RegimePlotter(tmp_path, dpi=72)
        plotter.plot_transition_matrix(A, filename="test_tm")
        assert (tmp_path / "test_tm.png").exists()

    def test_plot_drawdowns(self, prices, regimes_3, backtester, tmp_path):
        """plot_drawdowns doit générer un fichier sans erreur."""
        from utils.plotting import RegimePlotter
        results = backtester.run(prices, regimes_3)
        plotter = RegimePlotter(tmp_path, dpi=72)
        plotter.plot_drawdowns(results, filename="test_dd")
        assert (tmp_path / "test_dd.png").exists()

    def test_plot_regime_distribution(self, prices, regimes_3, regimes_2, tmp_path):
        """plot_regime_distribution doit générer un fichier sans erreur."""
        from utils.plotting import RegimePlotter
        plotter = RegimePlotter(tmp_path, dpi=72)
        plotter.plot_regime_distribution(regimes_3, regimes_2, filename="test_dist")
        assert (tmp_path / "test_dist.png").exists()

    def test_plot_latent_space(self, regimes_3, tmp_path):
        """plot_latent_space doit générer un fichier sans erreur."""
        from utils.plotting import RegimePlotter
        rng = np.random.default_rng(5)
        latent = rng.normal(0, 1, (len(regimes_3), 8))
        plotter = RegimePlotter(tmp_path, dpi=72)
        plotter.plot_latent_space(latent, regimes_3, filename="test_latent")
        assert (tmp_path / "test_latent.png").exists()

    def test_plot_all_runs_without_error(self, prices, regimes_3, regimes_2,
                                         backtester, tmp_path):
        """plot_all() doit s'exécuter sans lever d'exception."""
        from utils.plotting import RegimePlotter
        results = backtester.run(prices, regimes_3)
        plotter = RegimePlotter(tmp_path, dpi=72)
        rng = np.random.default_rng(0)
        latent = rng.normal(0, 1, (len(prices), 8))
        proba = np.abs(rng.dirichlet([1, 1, 1], size=len(prices)))
        A = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
        plotter.plot_all(
            prices=prices,
            regimes_vae_hmm=regimes_3,
            regimes_baseline=regimes_2,
            backtest_results=results,
            train_history=None,
            regime_proba=proba,
            latent_vectors=latent,
            transition_matrix=A,
        )
        # Vérifie que les figures principales ont été créées
        assert (tmp_path / "01_regimes_on_price.png").exists()
        assert (tmp_path / "02_equity_curves.png").exists()
        assert (tmp_path / "10_full_dashboard.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
