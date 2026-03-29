"""
config/constants.py
===================
Constantes globales du projet : labels de régimes, couleurs de visualisation,
noms de colonnes standardisés.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Labels et couleurs des régimes de marché
# ---------------------------------------------------------------------------

#: Mapping regime_id -> label lisible
REGIME_LABELS: dict[int, str] = {
    0: "Bear / Haute Volatilité",
    1: "Transition / Incertitude",
    2: "Bull / Basse Volatilité",
}

#: Couleurs matplotlib pour chaque régime (daltonien-friendly)
REGIME_COLORS: dict[int, str] = {
    0: "#E74C3C",  # rouge - bear
    1: "#F39C12",  # orange - transition
    2: "#27AE60",  # vert - bull
}

#: Couleurs pour le modèle de référence (2 régimes)
BASELINE_REGIME_LABELS: dict[int, str] = {
    0: "Régime Bas (bear)",
    1: "Régime Haut (bull)",
}
BASELINE_REGIME_COLORS: dict[int, str] = {
    0: "#E74C3C",
    1: "#27AE60",
}

# ---------------------------------------------------------------------------
# Noms de colonnes standardisés
# ---------------------------------------------------------------------------

#: Colonne de prix ajusté de clôture
COL_CLOSE: str = "Close"
COL_OPEN: str = "Open"
COL_HIGH: str = "High"
COL_LOW: str = "Low"
COL_VOLUME: str = "Volume"

#: Colonnes de features engineerées
COL_LOG_RETURN: str = "log_return"
COL_REALIZED_VOL: str = "realized_vol_{window}d"
COL_ROLLING_RETURN: str = "rolling_return_{window}d"
COL_RSI: str = "rsi"
COL_BB_WIDTH: str = "bb_width"
COL_BB_POSITION: str = "bb_position"
COL_TREND: str = "trend_50_200"

#: Colonne de régime prédit
COL_REGIME: str = "regime"
COL_REGIME_PROB: str = "regime_prob_{r}"

# ---------------------------------------------------------------------------
# Métriques de performance
# ---------------------------------------------------------------------------

METRICS_NAMES: list[str] = [
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "n_trades",
    "avg_trade_duration_days",
]

# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

MODEL_VAE_HMM: str = "VAE-HMM"
MODEL_MARKOV_SWITCHING: str = "Markov-Switching (Hamilton)"
MODEL_BUY_HOLD: str = "Buy & Hold"

TRADING_DAYS_PER_YEAR: int = 252
