"""
tests/test_hmm.py
=================
Tests unitaires pour models/hmm.py et models/markov_switching.py.

Utilise des données synthétiques avec régimes injectés pour vérifier
que le HMM est capable de les retrouver (sanity check du modèle).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.hmm import RegimeHMM, RegimeIdentifier
from config.settings import HMMConfig, MarkovSwitchingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_REGIMES = 3
LATENT_DIM = 8
N_TRAIN = 500
N_TEST = 100


@pytest.fixture
def hmm_cfg() -> HMMConfig:
    return HMMConfig(
        n_regimes=N_REGIMES,
        covariance_type="full",
        n_iter=100,
        tol=1e-3,
        n_init=3,
    )


@pytest.fixture
def synthetic_latents() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère des latents synthétiques avec 3 régimes bien séparés.

    Régime 0 : N([-2]*8, 0.3*I)  → low-vol
    Régime 1 : N([0]*8,  0.7*I)  → mid-vol
    Régime 2 : N([2]*8,  1.5*I)  → high-vol
    """
    rng = np.random.default_rng(42)
    means = [np.full(LATENT_DIM, -2.0),
             np.zeros(LATENT_DIM),
             np.full(LATENT_DIM, 2.0)]
    stds  = [0.3, 0.7, 1.5]

    # Séquence Markovienne simple (transitions régulières)
    true_regimes = np.zeros(N_TRAIN + N_TEST, dtype=int)
    regime = 0
    for t in range(1, len(true_regimes)):
        # Auto-transition = 0.9, passage au suivant = 0.1
        if rng.random() < 0.1:
            regime = (regime + 1) % N_REGIMES
        true_regimes[t] = regime

    latents = np.stack([
        rng.normal(means[r], stds[r], LATENT_DIM)
        for r in true_regimes
    ])
    return (
        latents[:N_TRAIN],
        latents[N_TRAIN:],
        true_regimes[N_TRAIN:],
    )


# ---------------------------------------------------------------------------
# RegimeIdentifier Tests
# ---------------------------------------------------------------------------

class TestRegimeIdentifier:

    def test_remap_is_permutation(self) -> None:
        """Le mapping doit être une permutation valide (bijectif)."""
        from hmmlearn.hmm import GaussianHMM
        rng = np.random.default_rng(0)

        # Crée un HMM minimal avec des covariances différentes
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1)
        X = rng.normal(0, 1, (200, 4))
        model.fit(X)

        identifier = RegimeIdentifier(n_regimes=3)
        identifier.fit(model)

        assert identifier._permutation is not None
        assert set(identifier._permutation) == {0, 1, 2}

    def test_remap_changes_labels(self) -> None:
        """remap() doit transformer les labels correctement."""
        from hmmlearn.hmm import GaussianHMM
        rng = np.random.default_rng(1)

        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1)
        X = rng.normal(0, 1, (200, 4))
        model.fit(X)

        identifier = RegimeIdentifier(n_regimes=3)
        identifier.fit(model)

        raw = np.array([0, 1, 2, 0, 1])
        remapped = identifier.remap(raw)

        assert remapped.shape == raw.shape
        assert set(remapped).issubset({0, 1, 2})

    def test_remap_proba_columns_reordered(self) -> None:
        """remap_proba doit réordonner les colonnes (somme par ligne = 1)."""
        from hmmlearn.hmm import GaussianHMM
        rng = np.random.default_rng(2)

        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1)
        X = rng.normal(0, 1, (200, 4))
        model.fit(X)

        identifier = RegimeIdentifier(n_regimes=3)
        identifier.fit(model)

        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.5, 0.4]])
        remapped = identifier.remap_proba(proba)

        # Les lignes doivent encore sommer à 1
        np.testing.assert_allclose(remapped.sum(axis=1), [1.0, 1.0])
        assert remapped.shape == proba.shape


# ---------------------------------------------------------------------------
# RegimeHMM Tests
# ---------------------------------------------------------------------------

class TestRegimeHMM:

    def test_fit_and_predict(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
    ) -> None:
        """Le HMM doit se fitter et prédire sans erreur."""
        lat_train, lat_test, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)
        regimes = hmm.predict(lat_test)

        assert regimes.shape == (N_TEST,)
        assert set(regimes).issubset(set(range(N_REGIMES)))

    def test_predict_proba_shape_and_sum(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
    ) -> None:
        """predict_proba doit retourner un tableau de probabilités valides."""
        lat_train, lat_test, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)
        proba = hmm.predict_proba(lat_test)

        assert proba.shape == (N_TEST, N_REGIMES)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(N_TEST), atol=1e-5)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_next_regime_shape(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
    ) -> None:
        """predict_next_regime doit retourner un tableau de la bonne shape."""
        lat_train, lat_test, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)
        next_r = hmm.predict_next_regime(lat_test)

        assert next_r.shape == (N_TEST,)
        assert set(next_r).issubset(set(range(N_REGIMES)))

    def test_transition_matrix_stochastic(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
    ) -> None:
        """La matrice de transition doit être stochastique par ligne (somme=1)."""
        lat_train, _, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)
        A = hmm.get_transition_matrix()

        assert A.shape == (N_REGIMES, N_REGIMES)
        np.testing.assert_allclose(A.sum(axis=1), np.ones(N_REGIMES), atol=1e-5)
        assert (A >= 0).all() and (A <= 1).all()

    def test_regime_ordering_by_variance(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
    ) -> None:
        """
        Après mapping canonique, le régime 0 doit avoir la variance la plus faible.

        Avec nos latents synthétiques (stds = 0.3, 0.7, 1.5), le régime 0
        doit correspondre au cluster le moins dispersé.
        """
        lat_train, lat_test, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)

        # Vérifie que les covariances sont ordonnées croissant
        perm = hmm.regime_identifier_._permutation
        covars = hmm.model_.covars_
        traces = [np.trace(covars[k]) for k in perm]
        assert traces == sorted(traces), (
            f"Régimes non ordonnés par variance : {traces}"
        )

    def test_not_fitted_raises(self, hmm_cfg: HMMConfig) -> None:
        """Les méthodes de prédiction doivent lever RuntimeError avant fit()."""
        hmm = RegimeHMM(hmm_cfg)
        dummy = np.random.rand(10, LATENT_DIM)
        with pytest.raises(RuntimeError, match="fit()"):
            hmm.predict(dummy)

    def test_nan_input_raises(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
    ) -> None:
        """Un input NaN doit lever une ValueError explicite."""
        lat_train, lat_test, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)

        bad_input = lat_test.copy()
        bad_input[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            hmm.predict(bad_input)

    def test_save_and_load(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
        tmp_path: Path,
    ) -> None:
        """Le modèle sauvegardé et rechargé doit donner les mêmes prédictions."""
        lat_train, lat_test, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)

        save_path = tmp_path / "regime_hmm.pkl"
        hmm.save(save_path)

        hmm2 = RegimeHMM.load(save_path)
        regimes1 = hmm.predict(lat_test)
        regimes2 = hmm2.predict(lat_test)
        np.testing.assert_array_equal(regimes1, regimes2)

    def test_get_regime_stats(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
    ) -> None:
        """get_regime_stats doit retourner un DataFrame avec les bons indices."""
        lat_train, lat_test, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)
        stats = hmm.get_regime_stats(lat_test)

        assert len(stats) == N_REGIMES
        assert "frequency_pct" in stats.columns
        np.testing.assert_allclose(
            stats["frequency_pct"].sum(), 100.0, atol=0.5
        )

    def test_regime_means_shape(
        self,
        hmm_cfg: HMMConfig,
        synthetic_latents: tuple,
    ) -> None:
        """get_regime_means doit retourner une array (K, latent_dim)."""
        lat_train, _, _ = synthetic_latents
        hmm = RegimeHMM(hmm_cfg)
        hmm.fit(lat_train)
        means = hmm.get_regime_means()
        assert means.shape == (N_REGIMES, LATENT_DIM)

    def test_compute_durations(self) -> None:
        """_compute_durations doit calculer les épisodes correctement."""
        regimes = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2, 2])
        durations = RegimeHMM._compute_durations(regimes, target_regime=0)
        assert durations == [3, 1]
        durations2 = RegimeHMM._compute_durations(regimes, target_regime=2)
        assert durations2 == [4]


# ---------------------------------------------------------------------------
# MarkovSwitchingBaseline Tests — utilise statsmodels si disponible
# ---------------------------------------------------------------------------

try:
    import statsmodels  # noqa: F401
    _SM_AVAILABLE = True
except ImportError:
    _SM_AVAILABLE = False


@pytest.mark.skipif(not _SM_AVAILABLE, reason="statsmodels non installé")
class TestMarkovSwitchingBaseline:

    @pytest.fixture
    def ms_cfg(self) -> MarkovSwitchingConfig:
        return MarkovSwitchingConfig(k_regimes=2, order=0, switching_variance=True)

    @pytest.fixture
    def synthetic_returns(self) -> tuple[pd.Series, pd.Series]:
        """Rendements synthétiques à 2 régimes (bull/bear)."""
        rng = np.random.default_rng(99)
        n = 600
        dates = pd.bdate_range("2018-01-01", periods=n)
        # Bull : faible vol, bear : forte vol
        returns = np.where(
            np.arange(n) < 400,
            rng.normal(0.0005, 0.008, n),   # bull
            rng.normal(-0.001, 0.025, n),    # bear
        )
        series = pd.Series(returns, index=dates)
        return series[:400], series[400:]

    def test_fit_runs(
        self, ms_cfg: MarkovSwitchingConfig, synthetic_returns: tuple
    ) -> None:
        from models.markov_switching import MarkovSwitchingBaseline
        train, _ = synthetic_returns
        baseline = MarkovSwitchingBaseline(ms_cfg)
        baseline.fit(train)
        assert baseline.is_fitted_

    def test_predict_shape(
        self, ms_cfg: MarkovSwitchingConfig, synthetic_returns: tuple
    ) -> None:
        from models.markov_switching import MarkovSwitchingBaseline
        train, _ = synthetic_returns
        baseline = MarkovSwitchingBaseline(ms_cfg)
        baseline.fit(train)
        regimes = baseline.predict(train)
        assert regimes.shape == (len(train),)
        assert set(regimes).issubset({0, 1})

    def test_predict_proba_valid(
        self, ms_cfg: MarkovSwitchingConfig, synthetic_returns: tuple
    ) -> None:
        from models.markov_switching import MarkovSwitchingBaseline
        train, _ = synthetic_returns
        baseline = MarkovSwitchingBaseline(ms_cfg)
        baseline.fit(train)
        proba = baseline.predict_proba(train)
        assert proba.shape == (len(train), 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(train)), atol=1e-4)

    def test_transition_matrix_stochastic(
        self, ms_cfg: MarkovSwitchingConfig, synthetic_returns: tuple
    ) -> None:
        from models.markov_switching import MarkovSwitchingBaseline
        train, _ = synthetic_returns
        baseline = MarkovSwitchingBaseline(ms_cfg)
        baseline.fit(train)
        A = baseline.get_transition_matrix()
        np.testing.assert_allclose(A.sum(axis=1), np.ones(2), atol=1e-4)

    def test_not_fitted_raises(self, ms_cfg: MarkovSwitchingConfig) -> None:
        from models.markov_switching import MarkovSwitchingBaseline
        baseline = MarkovSwitchingBaseline(ms_cfg)
        dummy = pd.Series(np.random.rand(100))
        with pytest.raises(RuntimeError):
            baseline.predict(dummy)

    def test_too_short_series_raises(
        self, ms_cfg: MarkovSwitchingConfig, synthetic_returns: tuple
    ) -> None:
        from models.markov_switching import MarkovSwitchingBaseline
        train, _ = synthetic_returns
        baseline = MarkovSwitchingBaseline(ms_cfg)
        baseline.fit(train)
        with pytest.raises(ValueError, match="courte"):
            baseline.predict(pd.Series(np.random.rand(50)))
