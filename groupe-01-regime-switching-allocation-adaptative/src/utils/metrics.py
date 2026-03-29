"""
utils/metrics.py
================
Calcul des métriques de performance financière.

Toutes les métriques sont vectorisées (NumPy) et sans dépendance externe
autre que NumPy/Pandas. Conçu pour être appelé par ``evaluation/comparator.py``
et ``strategy/backtester.py``.

Fonctions
---------
compute_metrics
    Calcule l'ensemble complet des métriques pour une série de returns.
annualized_return
    CAGR à partir d'une série de returns journaliers.
annualized_volatility
    Volatilité annualisée (std × √252).
sharpe_ratio
    Ratio de Sharpe annualisé.
sortino_ratio
    Ratio de Sortino (downside deviation uniquement).
max_drawdown
    Maximum Drawdown sur une equity curve.
calmar_ratio
    CAGR / |Max Drawdown|.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from config.constants import TRADING_DAYS_PER_YEAR


def compute_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """
    Calcule l'ensemble complet des métriques de performance.

    Parameters
    ----------
    returns : pd.Series
        Returns journaliers nets (log ou arithmétiques).
    risk_free_rate : float, optional
        Taux sans risque annualisé. Par défaut 0.04 (4%).
    annualization : int, optional
        Facteur d'annualisation. 252 pour données journalières.

    Returns
    -------
    Dict[str, float]
        Dictionnaire complet des métriques de performance.

    Examples
    --------
    >>> m = compute_metrics(returns_series, risk_free_rate=0.04)
    >>> m["sharpe_ratio"]
    1.23
    """
    returns = returns.dropna()
    if len(returns) < 2:
        logger.warning("Série de returns trop courte pour calculer les métriques.")
        return {k: 0.0 for k in [
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "n_trades", "avg_trade_duration_days",
        ]}

    rf_daily = risk_free_rate / annualization

    # Rendements et equity
    total_ret = float((1 + returns).prod() - 1)
    n_days = len(returns)
    years = n_days / annualization
    cagr = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0.0

    ann_vol = float(returns.std() * np.sqrt(annualization))

    # Sharpe
    excess = returns - rf_daily
    sr = float(excess.mean() / excess.std() * np.sqrt(annualization)) if excess.std() > 0 else 0.0

    # Sortino
    downside = returns[returns < rf_daily]
    downside_vol = float(downside.std() * np.sqrt(annualization)) if len(downside) > 1 else 1e-8
    sortino = float((cagr - risk_free_rate) / downside_vol) if downside_vol > 0 else 0.0

    # Max Drawdown
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    mdd = float(drawdowns.min())

    # Calmar
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-8 else 0.0

    # Win rate
    win_rate = float((returns > 0).mean())

    return {
        "total_return": total_ret,
        "annualized_return": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": mdd,
        "calmar_ratio": round(calmar, 4),
        "win_rate": win_rate,
        "n_trades": 0,              # mis à jour par le backtester
        "avg_trade_duration_days": 0.0,
    }


def annualized_return(
    returns: pd.Series, annualization: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calcule le CAGR (Compound Annual Growth Rate).

    Parameters
    ----------
    returns : pd.Series
        Returns journaliers.
    annualization : int
        Facteur d'annualisation.

    Returns
    -------
    float
        CAGR annualisé.
    """
    total = float((1 + returns.dropna()).prod() - 1)
    years = len(returns) / annualization
    return float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0


def annualized_volatility(
    returns: pd.Series, annualization: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calcule la volatilité annualisée.

    Parameters
    ----------
    returns : pd.Series
        Returns journaliers.
    annualization : int
        Facteur d'annualisation.

    Returns
    -------
    float
        Volatilité annualisée (écart-type × √annualization).
    """
    return float(returns.dropna().std() * np.sqrt(annualization))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calcule le Ratio de Sharpe annualisé.

    Parameters
    ----------
    returns : pd.Series
        Returns journaliers.
    risk_free_rate : float
        Taux sans risque annualisé.
    annualization : int
        Facteur d'annualisation.

    Returns
    -------
    float
        Ratio de Sharpe annualisé.
    """
    rf_daily = risk_free_rate / annualization
    excess = returns.dropna() - rf_daily
    return float(excess.mean() / excess.std() * np.sqrt(annualization)) if excess.std() > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    """
    Calcule le Maximum Drawdown.

    Parameters
    ----------
    returns : pd.Series
        Returns journaliers.

    Returns
    -------
    float
        Maximum Drawdown (valeur négative, ex: -0.35 = -35%).
    """
    equity = (1 + returns.dropna()).cumprod()
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    return float(drawdowns.min())


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.04,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calcule le Sharpe ratio glissant.

    Parameters
    ----------
    returns : pd.Series
        Returns journaliers.
    window : int
        Taille de la fenêtre glissante (en jours). Par défaut 63 (trimestre).
    risk_free_rate : float
        Taux sans risque annualisé.
    annualization : int
        Facteur d'annualisation.

    Returns
    -------
    pd.Series
        Sharpe ratio annualisé glissant.
    """
    rf_daily = risk_free_rate / annualization
    excess = returns - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(annualization)).rename("rolling_sharpe")
