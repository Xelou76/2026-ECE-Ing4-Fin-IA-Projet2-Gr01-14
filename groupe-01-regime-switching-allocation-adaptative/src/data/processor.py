"""
data/processor.py
=================
Orchestration du pipeline de données complet :
  téléchargement → feature engineering → normalisation → séquençage → split.

Classes
-------
DataBundle
    Dataclass immuable contenant toutes les données prêtes pour l'entraînement.
MarketDataProcessor
    Pipeline principal de préparation des données.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler

from config.settings import DataConfig
from data.downloader import MarketDataDownloader
from data.features import FeatureEngineer


@dataclass
class DataBundle:
    """
    Conteneur immuable de toutes les données préparées.

    Toutes les arrays NumPy sont C-contiguës et dtype float32
    pour la compatibilité directe avec PyTorch.

    Attributes
    ----------
    features_scaled : pd.DataFrame
        Feature matrix complète normalisée (RobustScaler sur train).
    sequences_train : np.ndarray
        Shape (N_train, seq_len, n_features) — entrée du VAE (train).
    sequences_val : np.ndarray
        Shape (N_val, seq_len, n_features) — entrée du VAE (validation).
    sequences_test : np.ndarray
        Shape (N_test, seq_len, n_features) — entrée du VAE (test).
    returns_train : pd.Series
        Log-rendements journaliers SPY (train) — pour la baseline HMM.
    returns_val : pd.Series
        Log-rendements journaliers SPY (val).
    returns_test : pd.Series
        Log-rendements journaliers SPY (test).
    prices_test : pd.Series
        Prix bruts SPY sur la période test — pour le backtest.
    prices_full : pd.DataFrame
        Prix bruts de tous les actifs sur toute la période.
    feature_names : list[str]
        Noms des colonnes de la feature matrix.
    scaler : RobustScaler
        Scaler ajusté uniquement sur le train set (pour inverse_transform).
    train_size : int
        Nombre de séquences dans le split train.
    val_size : int
        Nombre de séquences dans le split val.
    test_size : int
        Nombre de séquences dans le split test.
    n_features : int
        Dimension d'entrée du VAE (= nombre de features).
    seq_len : int
        Longueur des séquences temporelles.
    dates_test : pd.DatetimeIndex
        Dates correspondant aux séquences de test.
    """

    features_scaled: pd.DataFrame
    sequences_train: np.ndarray
    sequences_val: np.ndarray
    sequences_test: np.ndarray
    returns_train: pd.Series
    returns_val: pd.Series
    returns_test: pd.Series
    prices_test: pd.Series
    prices_full: pd.DataFrame
    feature_names: list[str]
    scaler: RobustScaler
    train_size: int = field(init=False)
    val_size: int = field(init=False)
    test_size: int = field(init=False)
    n_features: int = field(init=False)
    seq_len: int = field(init=False)
    dates_test: pd.DatetimeIndex = field(init=False)

    def __post_init__(self) -> None:
        self.train_size = len(self.sequences_train)
        self.val_size = len(self.sequences_val)
        self.test_size = len(self.sequences_test)
        self.n_features = self.sequences_train.shape[2]
        self.seq_len = self.sequences_train.shape[1]
        # Les dates correspondent aux derniers jours de chaque séquence
        n_total = len(self.features_scaled)
        n_train_end = self.train_size + self.seq_len - 1
        n_val_end = n_train_end + self.val_size
        self.dates_test = self.features_scaled.index[n_val_end:]


class MarketDataProcessor:
    """
    Pipeline complet : données brutes → DataBundle prêt pour le modèle.

    Étapes internes :
    1. Téléchargement des prix (avec cache)
    2. Feature engineering (rendements, vol, RSI, BB, tendance, corrélations)
    3. Split chronologique train/val/test (sans data leakage)
    4. Normalisation RobustScaler (ajusté sur train uniquement)
    5. Construction des séquences temporelles glissantes (sliding windows)

    Parameters
    ----------
    cfg : DataConfig
        Configuration de données (tickers, dates, fenêtres, ratios).
    cache_dir : Path, optional
        Répertoire de cache. Si None, utilise cfg defaults.

    Examples
    --------
    >>> from config.settings import DataConfig
    >>> processor = MarketDataProcessor(DataConfig())
    >>> bundle = processor.run()
    >>> bundle.sequences_train.shape
    (2100, 30, 24)
    """

    def __init__(
        self,
        cfg: DataConfig,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        cache_dir = cache_dir or Path("data/cache")
        self._downloader = MarketDataDownloader(cache_dir)
        self._feature_engineer = FeatureEngineer(
            vol_windows=cfg.vol_windows,
            return_windows=cfg.return_windows,
            rsi_window=cfg.rsi_window,
            bb_window=cfg.bb_window,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, force_refresh: bool = False) -> DataBundle:
        """
        Exécute le pipeline complet et retourne un DataBundle.

        Parameters
        ----------
        force_refresh : bool, optional
            Si True, re-télécharge même si le cache existe.

        Returns
        -------
        DataBundle
            Toutes les données prêtes pour l'entraînement et l'évaluation.
        """
        logger.info("MarketDataProcessor — démarrage du pipeline")

        # 1. Téléchargement
        prices = self._downloader.download(
            tickers=self.cfg.tickers,
            start_date=self.cfg.start_date,
            end_date=self.cfg.end_date,
            force_refresh=force_refresh,
        )

        # 2. Feature Engineering
        features_raw = self._feature_engineer.fit_transform(
            prices=prices,
            benchmark_col=self.cfg.tickers[0],  # Premier ticker = benchmark
        )

        # Aligne les prix sur les features (lookback a réduit l'index)
        prices = prices.loc[features_raw.index]

        # 3. Split temporel (strict : pas de mélange aléatoire)
        train_idx, val_idx, test_idx = self._temporal_split(features_raw)
        logger.info(
            f"Split — Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
        )

        # 4. Normalisation (fit sur TRAIN uniquement — pas de data leakage)
        scaler, features_scaled = self._normalize(features_raw, train_idx)

        # 5. Séquences glissantes
        seq_len = self.cfg.sequence_length
        seqs_train = self._build_sequences(
            features_scaled.iloc[: len(train_idx)].values, seq_len
        )
        seqs_val = self._build_sequences(
            features_scaled.iloc[
                len(train_idx): len(train_idx) + len(val_idx)
            ].values,
            seq_len,
        )
        seqs_test = self._build_sequences(
            features_scaled.iloc[len(train_idx) + len(val_idx):].values, seq_len
        )

        # 6. Log-rendements du benchmark pour la baseline
        benchmark = self.cfg.tickers[0]
        log_returns = np.log(prices[benchmark]).diff().dropna()

        # Intersection des index : log_returns démarre 1 jour après prices
        # (à cause du .diff()), il peut manquer certaines dates de train_idx
        ret_train = log_returns.reindex(train_idx).dropna()
        ret_val   = log_returns.reindex(val_idx).dropna()
        ret_test  = log_returns.reindex(test_idx).dropna()

        bundle = DataBundle(
            features_scaled=features_scaled,
            sequences_train=seqs_train.astype(np.float32),
            sequences_val=seqs_val.astype(np.float32),
            sequences_test=seqs_test.astype(np.float32),
            returns_train=ret_train,
            returns_val=ret_val,
            returns_test=ret_test,
            prices_test=prices.loc[test_idx, benchmark],
            prices_full=prices,
            feature_names=self._feature_engineer.feature_names,
            scaler=scaler,
        )

        logger.success(
            f"DataBundle prêt — "
            f"Séquences: train={bundle.train_size} | val={bundle.val_size} | test={bundle.test_size} | "
            f"Shape: ({bundle.seq_len}, {bundle.n_features})"
        )
        return bundle

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _temporal_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Split chronologique strict sans data leakage.

        Le split est fait sur les indices temporels, pas aléatoirement.
        Train → Val → Test dans l'ordre temporel.

        Returns
        -------
        tuple of DatetimeIndex
            (train_idx, val_idx, test_idx)
        """
        n = len(df)
        n_train = int(n * self.cfg.train_ratio)
        n_val = int(n * self.cfg.val_ratio)

        train_idx = df.index[:n_train]
        val_idx = df.index[n_train: n_train + n_val]
        test_idx = df.index[n_train + n_val:]

        return train_idx, val_idx, test_idx

    @staticmethod
    def _normalize(
        df: pd.DataFrame, train_idx: pd.DatetimeIndex
    ) -> tuple[RobustScaler, pd.DataFrame]:
        """
        Normalisation RobustScaler ajusté uniquement sur le train set.

        Utilise RobustScaler (médiane + IQR) plutôt que StandardScaler
        pour sa robustesse aux outliers financiers (crises, flash crashes).

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix brute complète.
        train_idx : pd.DatetimeIndex
            Index du split train — seule partition vue par le scaler.

        Returns
        -------
        tuple[RobustScaler, pd.DataFrame]
            (scaler ajusté, feature matrix normalisée)
        """
        scaler = RobustScaler()
        train_data = df.loc[train_idx]
        scaler.fit(train_data.values)

        scaled_values = scaler.transform(df.values)
        df_scaled = pd.DataFrame(
            scaled_values, index=df.index, columns=df.columns, dtype=np.float32
        )

        # Clip les outliers extrêmes post-normalisation (±10σ)
        df_scaled = df_scaled.clip(-10, 10)

        logger.debug(
            f"RobustScaler ajusté sur {len(train_idx)} observations. "
            f"Plage post-scaling : [{df_scaled.min().min():.2f}, {df_scaled.max().max():.2f}]"
        )
        return scaler, df_scaled

    @staticmethod
    def _build_sequences(
        data: np.ndarray, seq_len: int
    ) -> np.ndarray:
        """
        Construit des séquences glissantes (sliding window).

        À partir d'un array 2D (T, F), construit un array 3D
        (T - seq_len + 1, seq_len, F) de fenêtres temporelles.

        Chaque séquence X[i] = data[i : i + seq_len, :] contient
        ``seq_len`` pas de temps consécutifs.

        Parameters
        ----------
        data : np.ndarray
            Array 2D (T, n_features).
        seq_len : int
            Longueur de la fenêtre temporelle (lookback).

        Returns
        -------
        np.ndarray
            Array 3D (T - seq_len + 1, seq_len, n_features).
        """
        n, f = data.shape
        if n < seq_len:
            raise ValueError(
                f"Données insuffisantes : {n} observations < seq_len={seq_len}"
            )
        n_seqs = n - seq_len + 1
        # np.lib.stride_tricks pour une allocation mémoire efficace (vue, pas copie)
        shape = (n_seqs, seq_len, f)
        strides = (data.strides[0], data.strides[0], data.strides[1])
        sequences = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        # Copie explicite pour éviter les modifications inattendues via les strides
        return sequences.copy()

    def get_feature_summary(self, bundle: DataBundle) -> pd.DataFrame:
        """
        Retourne un résumé statistique des features pour inspection.

        Parameters
        ----------
        bundle : DataBundle
            DataBundle issu de ``run()``.

        Returns
        -------
        pd.DataFrame
            Statistiques descriptives des features normalisées.
        """
        return bundle.features_scaled.describe().T.round(4)
