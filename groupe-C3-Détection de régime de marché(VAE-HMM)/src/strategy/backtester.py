"""
strategy/backtester.py
======================
Moteur de backtest vectorisé pour la stratégie adaptative VAE-HMM.

La stratégie alloue entre 3 classes d'actifs (equity/bond/cash) selon
le régime de marché détecté par le pipeline VAE-HMM.

Conception du backtest
-----------------------
- Backtest **vectorisé** NumPy (pas de boucle Python par jour).
- Allocation J-1 → return J (exécution au close suivant, réaliste).
- Frais de transaction appliqués au turnover effectif (bps).
- Seuil de dérive : évite les micro-rebalancements (coût/bénéfice).
- Equity curve rebased à 100 pour la comparaison inter-stratégies.

Classes
-------
BacktestResult
    Dataclass contenant tous les résultats d'un backtest.
AdaptiveStrategyBacktester
    Moteur de backtest adaptatif basé sur les régimes VAE-HMM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.constants import (
    MODEL_BUY_HOLD,
    MODEL_MARKOV_SWITCHING,
    MODEL_VAE_HMM,
    REGIME_LABELS,
    TRADING_DAYS_PER_YEAR,
)
from config.settings import StrategyConfig
from utils.metrics import compute_metrics, rolling_sharpe


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Conteneur complet des résultats d'un backtest.

    Attributes
    ----------
    strategy_name : str
        Identifiant de la stratégie.
    equity_curve : pd.Series
        Valeur du portefeuille (rebased à 100). Index = DatetimeIndex.
    returns : pd.Series
        Returns journaliers nets du portefeuille.
    allocations : pd.DataFrame
        Allocations journalières par classe d'actif (equity, bond, cash).
    regimes : pd.Series
        Séquence de régimes utilisée pour l'allocation.
    turnover : pd.Series
        Turnover journalier (somme des |Δalloc|).
    transaction_costs_cum : pd.Series
        Coûts de transaction cumulatifs.
    metrics : Dict[str, float]
        Métriques de performance (Sharpe, MaxDD, CAGR...).
    rolling_sharpe_63d : pd.Series
        Sharpe ratio glissant sur 63 jours.
    n_rebalances : int
        Nombre de rebalancements effectifs.
    """

    strategy_name: str
    equity_curve: pd.Series
    returns: pd.Series
    allocations: pd.DataFrame
    regimes: pd.Series
    turnover: pd.Series
    transaction_costs_cum: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)
    rolling_sharpe_63d: pd.Series = field(default_factory=pd.Series)
    n_rebalances: int = 0


# ---------------------------------------------------------------------------
# AdaptiveStrategyBacktester
# ---------------------------------------------------------------------------

class AdaptiveStrategyBacktester:
    """
    Backteste la stratégie adaptative VAE-HMM et les benchmarks.

    La stratégie alloue le portefeuille entre equity (SPY), obligations (TLT)
    et cash selon le régime détecté à chaque instant. Le rebalancement
    n'est déclenché que si la dérive dépasse ``rebalance_threshold``.

    Parameters
    ----------
    cfg : StrategyConfig
        Configuration de la stratégie (allocations, frais, seuil).

    Examples
    --------
    >>> backtester = AdaptiveStrategyBacktester(cfg)
    >>> results = backtester.run(prices=prices_test, regimes=regimes_test)
    >>> results["VAE-HMM"].metrics["sharpe_ratio"]
    1.43
    """

    # Colonnes d'allocation
    _ALLOC_COLS: List[str] = ["equity", "bond", "cash"]

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        # Coût par rebalancement en fraction décimale (bps × 2 sens / 10_000)
        self._cost_rate = cfg.transaction_cost_bps / 10_000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.Series,
        regimes: np.ndarray,
        prices_bond: Optional[pd.Series] = None,
    ) -> Dict[str, BacktestResult]:
        """
        Exécute le backtest complet : stratégie adaptative + benchmarks.

        Simule 3 stratégies en parallèle :
        1. **VAE-HMM** : allocation adaptative selon les régimes.
        2. **Buy & Hold** : 100% equity, sans rebalancement.
        3. Retourne les 3 résultats dans un dictionnaire.

        Parameters
        ----------
        prices : pd.Series
            Prix de clôture ajustés du benchmark (ex: SPY).
            Index = DatetimeIndex aligné avec ``regimes``.
        regimes : np.ndarray
            Séquence de régimes prédits — shape (N,), valeurs ∈ {0, 1, 2}.
            Doit avoir la même longueur que ``prices``.
        prices_bond : pd.Series, optional
            Prix de l'actif obligataire (ex: TLT). Si None, simule
            un return constant de risk_free_rate/252 pour la partie bond.

        Returns
        -------
        Dict[str, BacktestResult]
            Clés : MODEL_VAE_HMM, MODEL_BUY_HOLD.
        """
        if len(prices) != len(regimes):
            raise ValueError(
                f"Longueurs incompatibles : prices={len(prices)}, regimes={len(regimes)}."
            )

        logger.info(
            f"Backtest — {len(prices)} jours | "
            f"coûts={self.cfg.transaction_cost_bps}bps | "
            f"seuil dérive={self.cfg.rebalance_threshold:.0%}"
        )

        # Returns journaliers de chaque actif
        equity_returns = prices.pct_change().fillna(0.0)
        bond_returns = (
            prices_bond.pct_change().fillna(0.0)
            if prices_bond is not None
            else pd.Series(
                self.cfg.risk_free_rate / TRADING_DAYS_PER_YEAR,
                index=prices.index,
            )
        )
        cash_returns = pd.Series(
            self.cfg.risk_free_rate / TRADING_DAYS_PER_YEAR,
            index=prices.index,
        )

        results: Dict[str, BacktestResult] = {}

        # 1. Stratégie VAE-HMM Adaptative
        results[MODEL_VAE_HMM] = self._run_adaptive(
            equity_returns=equity_returns,
            bond_returns=bond_returns,
            cash_returns=cash_returns,
            regimes=regimes,
            strategy_name=MODEL_VAE_HMM,
        )

        # 2. Buy & Hold (benchmark passif)
        results[MODEL_BUY_HOLD] = self._run_buy_and_hold(
            equity_returns=equity_returns,
            strategy_name=MODEL_BUY_HOLD,
        )

        self._log_summary(results)
        return results

    def run_baseline(
        self,
        prices: pd.Series,
        regimes_baseline: np.ndarray,
        prices_bond: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Exécute le backtest pour la baseline Markov-Switching (Hamilton).

        Utilise les mêmes allocations que la stratégie VAE-HMM mais
        avec les régimes du modèle de référence. Permet la comparaison
        directe sous les mêmes hypothèses de frais.

        Parameters
        ----------
        prices : pd.Series
            Prix de clôture du benchmark.
        regimes_baseline : np.ndarray
            Régimes prédits par la baseline — shape (N,), valeurs ∈ {0, 1}.
        prices_bond : pd.Series, optional
            Prix de l'actif obligataire.

        Returns
        -------
        BacktestResult
            Résultat de la stratégie basée sur Hamilton.
        """
        equity_returns = prices.pct_change().fillna(0.0)
        bond_returns = (
            prices_bond.pct_change().fillna(0.0)
            if prices_bond is not None
            else pd.Series(
                self.cfg.risk_free_rate / TRADING_DAYS_PER_YEAR,
                index=prices.index,
            )
        )
        cash_returns = pd.Series(
            self.cfg.risk_free_rate / TRADING_DAYS_PER_YEAR,
            index=prices.index,
        )

        # La baseline Hamilton a 2 régimes : mappe sur les allocations existantes
        # 0 → bear (allocation bear), 1 → bull (allocation bull = régime 2)
        baseline_map = {0: 0, 1: 2}  # 0=bear, 1=bull dans Hamilton
        remapped = np.array([baseline_map.get(int(r), 1) for r in regimes_baseline])

        return self._run_adaptive(
            equity_returns=equity_returns,
            bond_returns=bond_returns,
            cash_returns=cash_returns,
            regimes=remapped,
            strategy_name=MODEL_MARKOV_SWITCHING,
        )

    # ------------------------------------------------------------------
    # Private — Stratégie adaptative
    # ------------------------------------------------------------------

    def _run_adaptive(
        self,
        equity_returns: pd.Series,
        bond_returns: pd.Series,
        cash_returns: pd.Series,
        regimes: np.ndarray,
        strategy_name: str,
    ) -> BacktestResult:
        """Backtest vectorisé de la stratégie adaptative."""
        n = len(equity_returns)
        dates = equity_returns.index

        # 1. Allocations cibles par régime (array 2D : N × 3)
        target_allocs = self._regimes_to_allocations(regimes)  # (N, 3)

        # 2. Détection des rebalancements (seuil de dérive)
        #    Comparaison allocation[t] vs allocation[t-1]
        prev_allocs = np.vstack([target_allocs[:1], target_allocs[:-1]])
        drift = np.abs(target_allocs - prev_allocs).sum(axis=1)  # (N,)
        rebalance_mask = drift > self.cfg.rebalance_threshold
        rebalance_mask[0] = True  # Premier jour : toujours rebalancer

        # 3. Allocation effective : forward-fill vectorisé avec numba-style trick
        # np.where ne supporte pas le carry-forward → on utilise un scan cumulatif
        # sur les indices de rebalancement (méthode "last observation carried forward")
        effective_allocs = np.zeros_like(target_allocs)
        effective_allocs[0] = target_allocs[0]
        # Indices du dernier rebalancement : cumsum sur le masque booléen
        last_rebal_idx = np.where(
            rebalance_mask,
            np.arange(n),
            0,
        )
        # Propagation : chaque indice prend la valeur du dernier rebalancement
        # (équivalent pandas ffill mais sur un index entier)
        np.maximum.accumulate(last_rebal_idx, out=last_rebal_idx)
        effective_allocs = target_allocs[last_rebal_idx]

        # 4. Turnover et coûts (appliqués le jour du rebalancement)
        prev_eff = np.vstack([effective_allocs[:1], effective_allocs[:-1]])
        turnover = np.abs(effective_allocs - prev_eff).sum(axis=1)
        costs_daily = turnover * self._cost_rate

        # 5. Returns du portefeuille
        #    Allocation de J-1 appliquée au return de J (exécution next-close)
        alloc_lag = np.vstack([effective_allocs[:1], effective_allocs[:-1]])
        asset_returns = np.column_stack([
            equity_returns.values,
            bond_returns.values,
            cash_returns.values,
        ])
        port_returns_raw = (alloc_lag * asset_returns).sum(axis=1)
        port_returns = port_returns_raw - costs_daily

        port_returns_series = pd.Series(port_returns, index=dates)
        equity_curve = (1 + port_returns_series).cumprod() * 100.0

        # 6. Métriques
        metrics = compute_metrics(port_returns_series, self.cfg.risk_free_rate)
        metrics["n_trades"] = int(rebalance_mask.sum())

        rs = rolling_sharpe(port_returns_series, window=63,
                            risk_free_rate=self.cfg.risk_free_rate)

        allocs_df = pd.DataFrame(
            effective_allocs,
            index=dates,
            columns=self._ALLOC_COLS,
        )

        return BacktestResult(
            strategy_name=strategy_name,
            equity_curve=equity_curve,
            returns=port_returns_series,
            allocations=allocs_df,
            regimes=pd.Series(regimes, index=dates),
            turnover=pd.Series(turnover, index=dates),
            transaction_costs_cum=pd.Series(costs_daily, index=dates).cumsum(),
            metrics=metrics,
            rolling_sharpe_63d=rs,
            n_rebalances=int(rebalance_mask.sum()),
        )

    def _run_buy_and_hold(
        self,
        equity_returns: pd.Series,
        strategy_name: str,
    ) -> BacktestResult:
        """Backtest Buy-and-Hold (100% equity, sans frais)."""
        dates = equity_returns.index
        n = len(equity_returns)

        equity_curve = (1 + equity_returns).cumprod() * 100.0
        metrics = compute_metrics(equity_returns, self.cfg.risk_free_rate)

        allocs_df = pd.DataFrame(
            {"equity": 1.0, "bond": 0.0, "cash": 0.0},
            index=dates,
        )

        return BacktestResult(
            strategy_name=strategy_name,
            equity_curve=equity_curve,
            returns=equity_returns,
            allocations=allocs_df,
            regimes=pd.Series(np.zeros(n, dtype=int), index=dates),
            turnover=pd.Series(np.zeros(n), index=dates),
            transaction_costs_cum=pd.Series(np.zeros(n), index=dates),
            metrics=metrics,
            rolling_sharpe_63d=rolling_sharpe(equity_returns, 63, self.cfg.risk_free_rate),
            n_rebalances=0,
        )

    def _regimes_to_allocations(self, regimes: np.ndarray) -> np.ndarray:
        """
        Convertit une séquence de régimes en matrice d'allocations cibles.

        Implémentation vectorisée : construit une lookup table (K×3) et
        utilise l'indexation NumPy avancée — O(K) au lieu de O(N).

        Parameters
        ----------
        regimes : np.ndarray
            Régimes — shape (N,), valeurs ∈ {0, 1, 2}.

        Returns
        -------
        np.ndarray
            Allocations — shape (N, 3), colonnes : [equity, bond, cash].
        """
        # Lookup table : une ligne par régime possible
        n_regimes = max(self.cfg.regime_allocations.keys()) + 1
        default = {"equity": 0.33, "bond": 0.33, "cash": 0.34}
        lookup = np.array([
            [
                self.cfg.regime_allocations.get(k, default).get("equity", 0.0),
                self.cfg.regime_allocations.get(k, default).get("bond", 0.0),
                self.cfg.regime_allocations.get(k, default).get("cash", 0.0),
            ]
            for k in range(n_regimes)
        ], dtype=np.float64)  # shape (K, 3)

        # Indexation vectorisée : O(1) par rapport à une boucle Python O(N)
        clipped = np.clip(regimes.astype(int), 0, n_regimes - 1)
        return lookup[clipped]  # shape (N, 3)

    @staticmethod
    def _log_summary(results: Dict[str, BacktestResult]) -> None:
        """Log des métriques principales de toutes les stratégies."""
        logger.info("─" * 55)
        logger.info("  RÉSULTATS BACKTEST")
        logger.info("─" * 55)
        for name, res in results.items():
            m = res.metrics
            logger.success(
                f"  [{name}] "
                f"Sharpe={m.get('sharpe_ratio', 0):.3f} | "
                f"CAGR={m.get('annualized_return', 0):.1%} | "
                f"MaxDD={m.get('max_drawdown', 0):.1%} | "
                f"Sortino={m.get('sortino_ratio', 0):.3f}"
            )
        logger.info("─" * 55)
