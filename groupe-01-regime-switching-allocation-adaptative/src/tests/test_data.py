"""
tests/test_data.py
==================
Tests unitaires pour les modules data/.

Utilise des données synthétiques pour éviter les dépendances réseau.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.features import FeatureEngineer
from data.processor import MarketDataProcessor
from config.settings import DataConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """DataFrame de prix synthétiques (marche aléatoire géométrique)."""
    rng = np.random.default_rng(42)
    n = 600  # ~2.4 ans
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = rng.normal(0.0003, 0.012, (n, 4))  # 4 actifs
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(
        prices,
        index=dates,
        columns=["SPY", "TLT", "GLD", "VIX"],
    )


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    return FeatureEngineer(
        vol_windows=[5, 21],
        return_windows=[1, 5],
        rsi_window=14,
        bb_window=20,
    )


# ---------------------------------------------------------------------------
# FeatureEngineer Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineer:
    """Tests de la classe FeatureEngineer."""

    def test_output_shape(
        self, feature_engineer: FeatureEngineer, synthetic_prices: pd.DataFrame
    ) -> None:
        """La feature matrix doit avoir plus de colonnes que de tickers."""
        features = feature_engineer.fit_transform(synthetic_prices)
        assert features.shape[1] > synthetic_prices.shape[1]

    def test_no_nan_after_dropna(
        self, feature_engineer: FeatureEngineer, synthetic_prices: pd.DataFrame
    ) -> None:
        """Aucun NaN ne doit subsister après fit_transform."""
        features = feature_engineer.fit_transform(synthetic_prices)
        assert not features.isna().any().any()

    def test_feature_names_populated(
        self, feature_engineer: FeatureEngineer, synthetic_prices: pd.DataFrame
    ) -> None:
        """La liste feature_names doit être remplie après fit_transform."""
        feature_engineer.fit_transform(synthetic_prices)
        assert len(feature_engineer.feature_names) > 0

    def test_rsi_range(
        self, feature_engineer: FeatureEngineer, synthetic_prices: pd.DataFrame
    ) -> None:
        """Le RSI normalisé doit être dans [0, 1]."""
        features = feature_engineer.fit_transform(synthetic_prices)
        rsi_cols = [c for c in features.columns if c.startswith("rsi")]
        for col in rsi_cols:
            assert features[col].between(0, 1).all(), f"{col} hors [0,1]"

    def test_rolling_vol_positive(
        self, feature_engineer: FeatureEngineer, synthetic_prices: pd.DataFrame
    ) -> None:
        """Les volatilités réalisées doivent être positives."""
        features = feature_engineer.fit_transform(synthetic_prices)
        vol_cols = [c for c in features.columns if c.startswith("realized_vol")]
        for col in vol_cols:
            assert (features[col] >= 0).all(), f"{col} contient des valeurs négatives"

    def test_index_is_datetime(
        self, feature_engineer: FeatureEngineer, synthetic_prices: pd.DataFrame
    ) -> None:
        """L'index doit être un DatetimeIndex."""
        features = feature_engineer.fit_transform(synthetic_prices)
        assert isinstance(features.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# MarketDataProcessor Tests
# ---------------------------------------------------------------------------

class TestMarketDataProcessor:
    """Tests du pipeline MarketDataProcessor avec données mockées."""

    def _make_processor(self, tmp_path: Path) -> MarketDataProcessor:
        cfg = DataConfig(
            tickers=["SPY", "TLT", "GLD", "VIX"],
            start_date="2020-01-01",
            end_date="2022-12-31",
            train_ratio=0.7,
            val_ratio=0.15,
            vol_windows=[5, 21],
            return_windows=[1, 5],
            sequence_length=20,
        )
        return MarketDataProcessor(cfg, cache_dir=tmp_path)

    def test_bundle_shapes_consistent(
        self, synthetic_prices: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Les shapes du DataBundle doivent être cohérentes."""
        processor = self._make_processor(tmp_path)
        with patch.object(
            processor._downloader,
            "download",
            return_value=synthetic_prices,
        ):
            bundle = processor.run()

        assert bundle.sequences_train.ndim == 3
        assert bundle.sequences_val.ndim == 3
        assert bundle.sequences_test.ndim == 3
        assert bundle.sequences_train.shape[1] == 20   # seq_len
        assert bundle.sequences_train.shape[2] == bundle.n_features

    def test_no_data_leakage_scaler(
        self, synthetic_prices: pd.DataFrame, tmp_path: Path
    ) -> None:
        """
        Le scaler doit être ajusté sur le train uniquement.
        Test : les paramètres du scaler correspondent aux statistiques train.
        """
        processor = self._make_processor(tmp_path)
        with patch.object(
            processor._downloader,
            "download",
            return_value=synthetic_prices,
        ):
            bundle = processor.run()

        # Le scaler doit avoir été ajusté (center_ et scale_ non nuls)
        assert hasattr(bundle.scaler, "center_")
        assert hasattr(bundle.scaler, "scale_")
        assert len(bundle.scaler.center_) == bundle.n_features

    def test_dtype_float32(
        self, synthetic_prices: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Les séquences doivent être en float32 pour PyTorch."""
        processor = self._make_processor(tmp_path)
        with patch.object(
            processor._downloader,
            "download",
            return_value=synthetic_prices,
        ):
            bundle = processor.run()

        assert bundle.sequences_train.dtype == np.float32
        assert bundle.sequences_test.dtype == np.float32

    def test_build_sequences_shape(self) -> None:
        """Test unitaire de _build_sequences."""
        data = np.random.rand(100, 10)
        seqs = MarketDataProcessor._build_sequences(data, seq_len=20)
        assert seqs.shape == (81, 20, 10)  # (100 - 20 + 1, 20, 10)

    def test_build_sequences_values(self) -> None:
        """Les valeurs des séquences doivent correspondre aux données source."""
        data = np.arange(50).reshape(10, 5).astype(float)
        seqs = MarketDataProcessor._build_sequences(data, seq_len=3)
        np.testing.assert_array_equal(seqs[0], data[:3])
        np.testing.assert_array_equal(seqs[1], data[1:4])
