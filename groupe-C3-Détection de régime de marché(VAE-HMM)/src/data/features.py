"""
data/features.py
================
Feature Engineering enrichi pour la détection de régimes de marché (v2).

Nouveaux indicateurs ajoutés :
  - Moments statistiques glissants (skewness, kurtosis) : signaux de tail risk
  - MACD : capture les changements de tendance inter-horizons
  - Average True Range (ATR) : volatilité intraday (si OHLCV disponible)
  - Drawdown courant : proxy de stress du marché
  - Z-score de volatilité : régime de vol par rapport à la moyenne historique
  - Autocorrélation des rendements : détecte les régimes momentum vs mean-revert

Classes
-------
FeatureEngineer
    Transforme des prix bruts en feature matrix prête pour le VAE.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import ta
from loguru import logger


class FeatureEngineer:
    """
    Calcule et assemble les features pour la détection de régimes.

    v2 : ajout de moments d'ordre supérieur (skewness, kurtosis), MACD,
    drawdown courant, z-score de vol, et autocorrélations.

    Ces features enrichissent le signal donné au VAE et améliorent
    l'Information Coefficient du signal de régime.

    Parameters
    ----------
    vol_windows : list[int]
        Fenêtres pour la volatilité réalisée glissante.
    return_windows : list[int]
        Fenêtres pour les rendements glissants.
    rsi_window : int
        Période du RSI.
    bb_window : int
        Période des Bandes de Bollinger.
    """

    def __init__(
        self,
        vol_windows: list[int] = None,
        return_windows: list[int] = None,
        rsi_window: int = 14,
        bb_window: int = 20,
    ) -> None:
        self.vol_windows = vol_windows or [5, 21, 63]
        self.return_windows = return_windows or [1, 5, 21]
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        prices: pd.DataFrame,
        benchmark_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calcule l'intégralité des features (v2 enrichi).

        Pipeline :
          1. Log-rendements multi-horizons
          2. Volatilité réalisée glissante (annualisée)
          3. Moments d'ordre supérieur : skewness et kurtosis glissants [NOUVEAU]
          4. Rendements glissants cumulés (momentum)
          5. RSI
          6. Bandes de Bollinger
          7. Tendance EMA50/200
          8. Ratio de volatilité court/long
          9. MACD [NOUVEAU]
         10. Z-score de volatilité [NOUVEAU]
         11. Drawdown courant [NOUVEAU]
         12. Autocorrélation des rendements [NOUVEAU]
         13. Corrélations glissantes inter-actifs (multi-actifs)
        """
        benchmark = benchmark_col or prices.columns[0]
        if benchmark not in prices.columns:
            raise ValueError(f"Colonne benchmark '{benchmark}' introuvable.")

        logger.info(f"Feature engineering v2 — benchmark : {benchmark}")
        frames: list[pd.DataFrame] = []

        # 1. Log-rendements multi-horizons
        frames.append(self._compute_log_returns(prices))

        # 2. Volatilité réalisée glissante
        vol_df = self._compute_rolling_volatility(prices)
        frames.append(vol_df)

        # 3. Moments d'ordre supérieur [NOUVEAU]
        frames.append(self._compute_rolling_moments(prices))

        # 4. Rendements glissants cumulés (momentum)
        frames.append(self._compute_cumulative_returns(prices))

        # 5. RSI
        frames.append(self._compute_rsi(prices[benchmark]).to_frame())

        # 6. Bandes de Bollinger
        frames.append(self._compute_bollinger_bands(prices[benchmark]))

        # 7. Tendance EMA50/200
        frames.append(self._compute_trend(prices[benchmark]).to_frame())

        # 8. Ratio de volatilité court/long
        frames.append(self._compute_volatility_ratios(prices))

        # 9. MACD [NOUVEAU]
        frames.append(self._compute_macd(prices[benchmark]))

        # 10. Z-score de volatilité [NOUVEAU]
        frames.append(self._compute_vol_zscore(prices))

        # 11. Drawdown courant [NOUVEAU]
        frames.append(self._compute_drawdown(prices[benchmark]).to_frame())

        # 12. Autocorrélation des rendements [NOUVEAU]
        frames.append(self._compute_autocorrelation(prices[benchmark]))

        # 13. Corrélations glissantes inter-actifs
        if prices.shape[1] > 1:
            frames.append(self._compute_rolling_correlations(prices))

        # --- Assemblage ---
        features = pd.concat(frames, axis=1)

        # Suppression des colonnes trop lacunaires (>50% NaN)
        nan_ratio = features.isna().mean()
        cols_to_drop = nan_ratio[nan_ratio > 0.5].index.tolist()
        if cols_to_drop:
            logger.warning(
                f"  {len(cols_to_drop)} colonne(s) supprimées (>50% NaN) : "
                f"{cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}"
            )
            features = features.drop(columns=cols_to_drop)

        self._feature_names = list(features.columns)

        n_before = len(features)
        features = features.dropna()
        n_after = len(features)

        if n_after == 0:
            raise ValueError(
                "Aucune observation valide après suppression des NaN. "
                f"Vérifiez les tickers ({list(prices.columns)}) et les dates."
            )

        logger.info(
            f"Features v2 : {features.shape[1]} colonnes | "
            f"{n_before - n_after} lignes supprimées (lookback) | "
            f"{n_after} observations valides"
        )
        return features

    # ------------------------------------------------------------------
    # Feature Builders (originaux)
    # ------------------------------------------------------------------

    def _compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Log-rendements sur plusieurs horizons temporels."""
        frames = {}
        for ticker in prices.columns:
            for h in self.return_windows:
                frames[f"log_ret_{ticker}_{h}d"] = np.log(prices[ticker]).diff(h)
        return pd.DataFrame(frames, index=prices.index)

    def _compute_rolling_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Volatilité réalisée glissante annualisée σ_t^{(w)} = √252 × std(log_r)."""
        log_ret_1d = np.log(prices).diff()
        frames = {}
        for ticker in prices.columns:
            for w in self.vol_windows:
                frames[f"realized_vol_{ticker}_{w}d"] = (
                    log_ret_1d[ticker]
                    .rolling(w, min_periods=w // 2)
                    .std()
                    * np.sqrt(252)
                )
        return pd.DataFrame(frames, index=prices.index)

    def _compute_cumulative_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Rendements simples glissants cumulés (momentum)."""
        frames = {}
        for ticker in prices.columns:
            for h in self.return_windows:
                if h == 1:
                    continue
                frames[f"momentum_{ticker}_{h}d"] = (
                    prices[ticker].ffill().pct_change(h, fill_method=None)
                )
        return pd.DataFrame(frames, index=prices.index)

    def _compute_rsi(self, series: pd.Series) -> pd.Series:
        """RSI normalisé sur [0, 1]."""
        rsi = ta.momentum.RSIIndicator(
            close=series, window=self.rsi_window, fillna=False
        )
        result = rsi.rsi()
        result.name = f"rsi_{self.rsi_window}"
        return result / 100.0

    def _compute_bollinger_bands(self, series: pd.Series) -> pd.DataFrame:
        """Bandes de Bollinger : largeur et position."""
        bb = ta.volatility.BollingerBands(
            close=series, window=self.bb_window, window_dev=2, fillna=False
        )
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        middle = bb.bollinger_mavg()

        width = (upper - lower) / middle.replace(0, np.nan)
        position = (series - lower) / (upper - lower).replace(0, np.nan)

        return pd.DataFrame(
            {
                f"bb_width_{self.bb_window}": width,
                f"bb_position_{self.bb_window}": position,
            },
            index=series.index,
        )

    def _compute_trend(self, series: pd.Series) -> pd.Series:
        """Signal de tendance EMA50 vs EMA200."""
        ema_fast = series.ewm(span=50, adjust=False).mean()
        ema_slow = series.ewm(span=200, adjust=False).mean()
        trend = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
        trend.name = "trend_ema50_200"
        return trend

    def _compute_volatility_ratios(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Ratio de volatilité court/long (signal précurseur de changement de régime)."""
        frames = {}
        log_ret_1d = np.log(prices).diff()
        for ticker in prices.columns:
            short_vol = log_ret_1d[ticker].rolling(5, min_periods=3).std() * np.sqrt(252)
            long_vol = log_ret_1d[ticker].rolling(63, min_periods=30).std() * np.sqrt(252)
            frames[f"vol_ratio_{ticker}"] = short_vol / long_vol.replace(0, np.nan)
        return pd.DataFrame(frames, index=prices.index)

    def _compute_rolling_correlations(
        self, prices: pd.DataFrame, window: int = 63
    ) -> pd.DataFrame:
        """Corrélations glissantes entre paires d'actifs (flight-to-quality)."""
        log_ret = np.log(prices).diff()
        cols = list(prices.columns)
        frames = {}
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if j <= i:
                    continue
                frames[f"corr_{c1}_{c2}_{window}d"] = (
                    log_ret[[c1, c2]]
                    .rolling(window, min_periods=window // 2)
                    .corr()
                    .unstack()[c1][c2]
                )
        return pd.DataFrame(frames, index=prices.index)

    # ------------------------------------------------------------------
    # Nouvelles features v2
    # ------------------------------------------------------------------

    def _compute_rolling_moments(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Skewness et kurtosis glissants des log-rendements.

        Interprétation pour les régimes :
        - Skewness fortement négative → queue gauche épaisse → risque de crash (bear)
        - Kurtosis élevée → fat tails → régime de stress / haute volatilité
        - Skewness positive, kurtosis normale → régime bull calme

        Fenêtre choisie : 63 jours (trimestrielle) pour capturer des régimes
        d'une durée réaliste sans trop de bruit.
        """
        log_ret = np.log(prices).diff()
        frames = {}
        window = 63

        for ticker in prices.columns:
            s = log_ret[ticker]

            # Skewness glissante (3e moment centré normalisé)
            frames[f"skewness_{ticker}_{window}d"] = (
                s.rolling(window, min_periods=window // 2).skew()
            )

            # Kurtosis glissante (4e moment → mesure de fat tails)
            # kurtosis(normale) = 0 (excess kurtosis via pandas)
            frames[f"kurtosis_{ticker}_{window}d"] = (
                s.rolling(window, min_periods=window // 2).kurt()
            )

        return pd.DataFrame(frames, index=prices.index)

    def _compute_macd(self, series: pd.Series) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence).

        Captures les changements de momentum via la différence entre EMA rapide
        et lente, normalisée par le prix.

        Features extraites :
        - macd_line : (EMA12 - EMA26) / prix → signal de tendance
        - macd_signal : EMA9 du macd_line → signal de croisement
        - macd_histogram : macd_line - macd_signal → force du signal

        Interprétation :
        - Histogramme > 0 et croissant → momentum haussier (bull)
        - Histogramme < 0 et décroissant → momentum baissier (bear)
        """
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = (ema12 - ema26) / series.replace(0, np.nan)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            },
            index=series.index,
        )

    def _compute_vol_zscore(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score de volatilité : volatilité courante vs sa moyenne historique.

        z_vol_t = (vol_t^{(21)} - mean(vol^{(21)}_{t-252:t})) / std(vol^{(21)}_{t-252:t})

        Un z-score élevé (> 1.5) indique une volatilité anormalement haute
        → signal fort de régime de stress / transition bear.

        C'est l'un des indicateurs les plus fiables pour l'IC des régimes,
        car il compare la volatilité à son propre historique récent.
        """
        log_ret = np.log(prices).diff()
        frames = {}

        for ticker in prices.columns:
            vol_21 = log_ret[ticker].rolling(21, min_periods=10).std() * np.sqrt(252)
            # Z-score sur une fenêtre d'un an
            vol_mean = vol_21.rolling(252, min_periods=126).mean()
            vol_std = vol_21.rolling(252, min_periods=126).std()
            frames[f"vol_zscore_{ticker}"] = (vol_21 - vol_mean) / vol_std.replace(0, np.nan)

        return pd.DataFrame(frames, index=prices.index)

    def _compute_drawdown(self, series: pd.Series) -> pd.Series:
        """
        Drawdown courant depuis le plus haut récent (252 jours glissants).

        drawdown_t = (P_t - max(P_{t-252:t})) / max(P_{t-252:t})

        Le drawdown est un indicateur direct du régime bear :
        - drawdown > -5% → régime bull ou transition
        - drawdown < -15% → régime bear prononcé

        Normalisé pour être dimensionless (entre -1 et 0).
        """
        rolling_max = series.rolling(252, min_periods=63).max()
        drawdown = (series - rolling_max) / rolling_max.replace(0, np.nan)
        drawdown.name = "drawdown_252d"
        return drawdown

    def _compute_autocorrelation(
        self, series: pd.Series, lags: list[int] = None
    ) -> pd.DataFrame:
        """
        Autocorrélation glissante des log-rendements à différents lags.

        autocorr_t^{(lag)} = corr(r_{t-w:t}, r_{t-w-lag:t-lag})

        Interprétation pour les régimes :
        - Autocorrélation positive (lag 1, 5) → trend following / momentum → bull
        - Autocorrélation négative → mean reversion → souvent régime bear ou consolidation
        - Autocorrélation ~0 → marché aléatoire / faible signal

        Les études montrent que les autocorrélations changent significativement
        entre régimes bull et bear (Lo, 1991 ; Ang & Bekaert, 2004).
        """
        lags = lags or [1, 5]
        log_ret = np.log(series).diff()
        frames = {}
        window = 63

        for lag in lags:
            autocorr_series = pd.Series(index=series.index, dtype=float)

            for t in range(window + lag, len(log_ret)):
                window_data = log_ret.iloc[t - window:t]
                lagged_data = log_ret.iloc[t - window - lag:t - lag]
                if len(window_data) > 10 and len(lagged_data) > 10:
                    corr = window_data.corr(lagged_data)
                    autocorr_series.iloc[t] = corr

            frames[f"autocorr_{lag}d_{window}w"] = autocorr_series

        return pd.DataFrame(frames, index=series.index)