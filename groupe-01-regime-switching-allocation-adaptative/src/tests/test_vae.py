"""
tests/test_vae.py
=================
Tests unitaires pour models/vae.py et models/trainer.py.

Utilise uniquement le CPU et des données synthétiques.
Aucune dépendance réseau ou GPU requise.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from models.vae import LSTMDecoder, LSTMEncoder, TimeSeriesVAE, VAEOutput
from models.trainer import EarlyStopping, KLScheduler, TrainingHistory, VAETrainer
from config.settings import VAEConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH, SEQ_LEN, INPUT_DIM, HIDDEN_DIM, LATENT_DIM = 16, 20, 12, 64, 4


@pytest.fixture
def model() -> TimeSeriesVAE:
    return TimeSeriesVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        seq_len=SEQ_LEN,
        num_layers=2,
        dropout=0.0,  # désactivé pour les tests
    )


@pytest.fixture
def x_batch() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(BATCH, SEQ_LEN, INPUT_DIM)


@pytest.fixture
def vae_cfg(tmp_path: Path) -> VAEConfig:
    return VAEConfig(
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=2,
        dropout=0.0,
        epochs=3,
        batch_size=8,
        learning_rate=1e-3,
        beta_warmup_epochs=2,
        early_stopping_patience=5,
    )


@pytest.fixture
def synthetic_bundle():
    """DataBundle synthétique minimal."""
    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 80, 20, 20
    shape = (1, SEQ_LEN, INPUT_DIM)
    bundle = MagicMock()
    bundle.sequences_train = rng.random((n_train, SEQ_LEN, INPUT_DIM)).astype(np.float32)
    bundle.sequences_val   = rng.random((n_val, SEQ_LEN, INPUT_DIM)).astype(np.float32)
    bundle.sequences_test  = rng.random((n_test, SEQ_LEN, INPUT_DIM)).astype(np.float32)
    bundle.n_features = INPUT_DIM
    bundle.seq_len = SEQ_LEN
    return bundle


# ---------------------------------------------------------------------------
# LSTMEncoder Tests
# ---------------------------------------------------------------------------

class TestLSTMEncoder:

    def test_output_shapes(self, x_batch: torch.Tensor) -> None:
        encoder = LSTMEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, num_layers=2)
        mu, log_var = encoder(x_batch)
        assert mu.shape == (BATCH, LATENT_DIM)
        assert log_var.shape == (BATCH, LATENT_DIM)

    def test_log_var_clamped(self, x_batch: torch.Tensor) -> None:
        """log_var doit rester dans [-10, 4] pour la stabilité numérique."""
        encoder = LSTMEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        _, log_var = encoder(x_batch)
        assert (log_var >= -10.0).all()
        assert (log_var <= 4.0).all()

    def test_deterministic_eval(self, x_batch: torch.Tensor) -> None:
        """En eval mode, deux passes identiques → mêmes sorties."""
        encoder = LSTMEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
        encoder.eval()
        with torch.no_grad():
            mu1, lv1 = encoder(x_batch)
            mu2, lv2 = encoder(x_batch)
        torch.testing.assert_close(mu1, mu2)
        torch.testing.assert_close(lv1, lv2)


# ---------------------------------------------------------------------------
# LSTMDecoder Tests
# ---------------------------------------------------------------------------

class TestLSTMDecoder:

    def test_output_shape(self) -> None:
        decoder = LSTMDecoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM, SEQ_LEN, num_layers=2)
        z = torch.randn(BATCH, LATENT_DIM)
        x_recon = decoder(z)
        assert x_recon.shape == (BATCH, SEQ_LEN, INPUT_DIM)

    def test_gradient_flows(self) -> None:
        """Les gradients doivent se propager jusqu'aux paramètres du décodeur."""
        decoder = LSTMDecoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM, SEQ_LEN)
        z = torch.randn(BATCH, LATENT_DIM, requires_grad=True)
        x_recon = decoder(z)
        loss = x_recon.mean()
        loss.backward()
        assert z.grad is not None


# ---------------------------------------------------------------------------
# TimeSeriesVAE Tests
# ---------------------------------------------------------------------------

class TestTimeSeriesVAE:

    def test_forward_output_types(
        self, model: TimeSeriesVAE, x_batch: torch.Tensor
    ) -> None:
        """Le forward pass doit retourner un VAEOutput valide."""
        model.train()
        out = model(x_batch, beta=0.5)
        assert isinstance(out, VAEOutput)
        assert isinstance(out.elbo, torch.Tensor)
        assert out.elbo.requires_grad

    def test_reconstruction_shape(
        self, model: TimeSeriesVAE, x_batch: torch.Tensor
    ) -> None:
        """x_recon doit avoir la même shape que x."""
        out = model(x_batch)
        assert out.x_recon.shape == x_batch.shape

    def test_latent_shape(
        self, model: TimeSeriesVAE, x_batch: torch.Tensor
    ) -> None:
        """Le vecteur latent z doit avoir la bonne dimension."""
        out = model(x_batch)
        assert out.z.shape == (BATCH, LATENT_DIM)
        assert out.mu.shape == (BATCH, LATENT_DIM)

    def test_kl_loss_non_negative(
        self, model: TimeSeriesVAE, x_batch: torch.Tensor
    ) -> None:
        """La KL divergence KL(q||p) doit être >= 0."""
        model.train()
        out = model(x_batch, beta=1.0)
        assert out.kl_loss.item() >= -1e-4  # tolérance numérique

    def test_elbo_decomposes_correctly(
        self, model: TimeSeriesVAE, x_batch: torch.Tensor
    ) -> None:
        """ELBO = recon_loss + β × kl_loss."""
        beta = 0.7
        model.train()
        out = model(x_batch, beta=beta)
        expected = out.recon_loss + beta * out.kl_loss
        torch.testing.assert_close(out.elbo, expected)

    def test_beta_zero_ignores_kl(
        self, model: TimeSeriesVAE, x_batch: torch.Tensor
    ) -> None:
        """Avec β=0, l'ELBO == recon_loss (KL ignorée)."""
        model.train()
        out = model(x_batch, beta=0.0)
        torch.testing.assert_close(out.elbo, out.recon_loss)

    def test_reparameterize_train_vs_eval(
        self, model: TimeSeriesVAE
    ) -> None:
        """En eval, reparameterize doit retourner μ directement (déterministe)."""
        mu = torch.zeros(BATCH, LATENT_DIM)
        log_var = torch.zeros(BATCH, LATENT_DIM)

        model.eval()
        z_eval = model.reparameterize(mu, log_var)
        torch.testing.assert_close(z_eval, mu)

    def test_encode_is_deterministic(
        self, model: TimeSeriesVAE, x_batch: torch.Tensor
    ) -> None:
        """model.encode() doit être déterministe (mode eval)."""
        z1 = model.encode(x_batch)
        z2 = model.encode(x_batch)
        torch.testing.assert_close(z1, z2)

    def test_n_params_positive(self, model: TimeSeriesVAE) -> None:
        assert model.n_params > 0

    def test_sample_shape(self, model: TimeSeriesVAE) -> None:
        """model.sample() doit générer des séquences de la bonne shape."""
        samples = model.sample(n_samples=5, device=torch.device("cpu"))
        assert samples.shape == (5, SEQ_LEN, INPUT_DIM)

    def test_backward_pass(
        self, model: TimeSeriesVAE, x_batch: torch.Tensor
    ) -> None:
        """Le backward pass complet ne doit pas lever d'exception."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        out = model(x_batch, beta=0.5)
        out.elbo.backward()
        optimizer.step()
        # Vérifie que les gradients ont été calculés
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient manquant : {name}"


# ---------------------------------------------------------------------------
# KLScheduler Tests
# ---------------------------------------------------------------------------

class TestKLScheduler:

    def test_starts_at_beta_start(self) -> None:
        sched = KLScheduler(beta_start=0.0, beta_end=1.0, warmup_epochs=10)
        assert sched.beta == 0.0

    def test_linear_progression(self) -> None:
        sched = KLScheduler(beta_start=0.0, beta_end=1.0, warmup_epochs=10)
        betas = []
        for _ in range(15):
            betas.append(sched.beta)
            sched.step()
        # Après warmup_epochs, doit être clamped à beta_end
        assert betas[10] == 1.0
        assert betas[14] == 1.0

    def test_monotone_increase(self) -> None:
        sched = KLScheduler(beta_start=0.0, beta_end=1.0, warmup_epochs=5)
        betas = []
        for _ in range(7):
            betas.append(sched.beta)
            sched.step()
        # Doit être non-décroissant
        assert all(b1 <= b2 for b1, b2 in zip(betas, betas[1:]))


# ---------------------------------------------------------------------------
# EarlyStopping Tests
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_stops_after_patience(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(1.0)  # meilleur
        es.step(1.1)  # pas d'amélioration
        es.step(1.2)
        assert not es.should_stop
        es.step(1.3)
        assert es.should_stop

    def test_resets_on_improvement(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(1.1)
        es.step(0.9)  # amélioration → reset
        es.step(1.0)
        assert not es.should_stop

    def test_returns_true_on_best(self) -> None:
        es = EarlyStopping(patience=5)
        assert es.step(1.0)   # première valeur → meilleur
        assert not es.step(1.1)  # pire → pas de checkpoint


# ---------------------------------------------------------------------------
# VAETrainer Integration Tests
# ---------------------------------------------------------------------------

class TestVAETrainer:

    def test_train_runs_without_error(
        self, vae_cfg: VAEConfig, synthetic_bundle, tmp_path: Path
    ) -> None:
        """L'entraînement sur 3 epochs doit se terminer sans exception."""
        trainer = VAETrainer(vae_cfg, tmp_path, device="cpu")
        model, history = trainer.train(synthetic_bundle)
        assert isinstance(history, TrainingHistory)
        assert len(history.train_loss) > 0

    def test_history_lengths_consistent(
        self, vae_cfg: VAEConfig, synthetic_bundle, tmp_path: Path
    ) -> None:
        """Toutes les listes d'historique doivent avoir la même longueur."""
        trainer = VAETrainer(vae_cfg, tmp_path, device="cpu")
        _, history = trainer.train(synthetic_bundle)
        lengths = {
            len(history.train_loss), len(history.val_loss),
            len(history.beta_values), len(history.learning_rates),
        }
        assert len(lengths) == 1  # toutes égales

    def test_checkpoint_saved(
        self, vae_cfg: VAEConfig, synthetic_bundle, tmp_path: Path
    ) -> None:
        """Le fichier vae_best.pt doit exister après l'entraînement."""
        trainer = VAETrainer(vae_cfg, tmp_path, device="cpu")
        trainer.train(synthetic_bundle)
        assert (tmp_path / "vae_best.pt").exists()

    def test_encode_all_shapes(
        self, vae_cfg: VAEConfig, synthetic_bundle, tmp_path: Path
    ) -> None:
        """encode_all doit retourner 3 arrays de la bonne shape."""
        trainer = VAETrainer(vae_cfg, tmp_path, device="cpu")
        model, _ = trainer.train(synthetic_bundle)
        lt, lv, lte = trainer.encode_all(model, synthetic_bundle)
        assert lt.shape == (80, LATENT_DIM)
        assert lv.shape == (20, LATENT_DIM)
        assert lte.shape == (20, LATENT_DIM)

    def test_load_checkpoint(
        self, vae_cfg: VAEConfig, synthetic_bundle, tmp_path: Path
    ) -> None:
        """Un modèle chargé depuis checkpoint doit donner les mêmes prédictions."""
        trainer = VAETrainer(vae_cfg, tmp_path, device="cpu")
        model1, _ = trainer.train(synthetic_bundle)
        model2 = trainer.load()

        x = torch.from_numpy(synthetic_bundle.sequences_test[:4])
        model1.eval(); model2.eval()
        with torch.no_grad():
            z1 = model1.encode(x)
            z2 = model2.encode(x)
        torch.testing.assert_close(z1, z2)
