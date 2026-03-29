"""
models/trainer.py
=================
Boucle d'entraînement du LSTM-VAE avec :
  - KL Annealing (β-schedule linéaire)
  - Early Stopping sur la RECONSTRUCTION loss (et non l'ELBO)
    → l'ELBO monte mécaniquement pendant le warmup (β augmente),
      ce qui déclenchait l'arrêt prématurément (best_epoch=1).
  - Gel de l'early stopping pendant la phase de warmup KL
  - Checkpointing (meilleur modèle sur val_recon)
  - Logging structuré des métriques par epoch
  - Extraction des représentations latentes (encode_all)

Corrections v3 :
  - EarlyStopping surveille val_recon au lieu de val_elbo (Fix #1 critique)
  - early_stopping.freeze() désactive le compteur pendant le KL warmup
  - Checkpoint sauvegardé sur val_recon (métrique stable)
  - Log du signal de monitoring explicite

Classes
-------
TrainingHistory
    Dataclass des courbes d'apprentissage.
EarlyStopping
    Callback d'arrêt précoce avec patience, gel de phase warmup.
KLScheduler
    Gère le β-schedule pour le KL Annealing.
VAETrainer
    Orchestre l'entraînement, la validation et l'évaluation du VAE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from config.settings import VAEConfig
from data.processor import DataBundle
from models.vae import TimeSeriesVAE, VAEOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """
    Historique complet de l'entraînement pour diagnostic et visualisation.

    Attributes
    ----------
    train_loss : list[float]
        ELBO totale sur le train set, par epoch.
    val_loss : list[float]
        ELBO totale sur le val set, par epoch.
    train_recon : list[float]
        Reconstruction loss (MSE) train, par epoch.
    val_recon : list[float]
        Reconstruction loss (MSE) val, par epoch.
    train_kl : list[float]
        KL divergence train (non pondérée), par epoch.
    val_kl : list[float]
        KL divergence val (non pondérée), par epoch.
    beta_values : list[float]
        Valeur de β (KL weight) à chaque epoch.
    learning_rates : list[float]
        Learning rate à chaque epoch.
    best_epoch : int
        Epoch du meilleur modèle (selon val_recon).
    """

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_recon: List[float] = field(default_factory=list)
    val_recon: List[float] = field(default_factory=list)
    train_kl: List[float] = field(default_factory=list)
    val_kl: List[float] = field(default_factory=list)
    beta_values: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0

    def to_dict(self) -> Dict[str, list]:
        """Sérialise l'historique en dict JSON-compatible."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_recon": self.train_recon,
            "val_recon": self.val_recon,
            "train_kl": self.train_kl,
            "val_kl": self.val_kl,
            "beta_values": self.beta_values,
            "learning_rates": self.learning_rates,
            "best_epoch": self.best_epoch,
        }

    def save(self, path: Path) -> None:
        """Sauvegarde l'historique en JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingHistory":
        """Charge l'historique depuis un fichier JSON."""
        with open(path) as f:
            data = json.load(f)
        h = cls()
        for k, v in data.items():
            setattr(h, k, v)
        return h


class EarlyStopping:
    """
    Callback d'arrêt précoce basé sur la RECONSTRUCTION loss.

    CORRECTION v3 : surveille val_recon et non val_elbo.
    Raison : l'ELBO = recon + beta*KL augmente mecaniquement pendant le
    KL warmup (beta croit a chaque epoch), ce qui declenchait l'arret
    prematurement (best_epoch=1 sur 250 planifies, beta max=0.375 au lieu
    de 1.0). La reconstruction loss est stable et reflete fidelement
    la qualite d'encodage independamment du schedule beta.

    Gel pendant le warmup (freeze/unfreeze) : pendant la phase de KL
    annealing, le modele est encore en train d'apprendre a structurer
    l'espace latent. Geler l'early stopping sur cette periode evite
    d'arreter avant que le modele ait pu converger correctement.

    Parameters
    ----------
    patience : int
        Nombre d'epochs sans amelioration avant l'arret.
    min_delta : float, optional
        Amelioration minimale requise pour reinitialiser le compteur.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter: int = 0
        self._best_loss: float = float("inf")
        self._frozen: bool = False  # Gel pendant le KL warmup

    @property
    def should_stop(self) -> bool:
        """True si on doit arrêter l'entraînement."""
        return (not self._frozen) and (self._counter >= self.patience)

    def freeze(self) -> None:
        """Gèle l'early stopping (pendant le warmup KL)."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Dégèle l'early stopping et remet le compteur à zéro."""
        self._frozen = False
        self._counter = 0       # Reset propre : évite un arrêt immédiat post-warmup
        self._best_loss = float("inf")  # Repart de zéro post-warmup — le premier
                                        # checkpoint sera forcément le modèle à β final

    def step(self, val_recon: float) -> bool:
        """
        Met à jour le compteur sur la reconstruction loss.

        Parameters
        ----------
        val_recon : float
            Validation RECONSTRUCTION loss de l'epoch courante.
            NB : ne pas passer val_elbo — ce serait l'ancien bug.

        Returns
        -------
        bool
            True si c'est le meilleur modèle jusqu'ici.
        """
        # Si gelé (warmup) : on suit le meilleur loss MAIS on ne signal pas is_best.
        # Le checkpoint sera sauvegardé à la fin du warmup, pas pendant.
        if self._frozen:
            if val_recon < self._best_loss - self.min_delta:
                self._best_loss = val_recon
            return False  # jamais de checkpoint pendant le warmup

        if val_recon < self._best_loss - self.min_delta:
            self._best_loss = val_recon
            self._counter = 0
            return True
        else:
            self._counter += 1
            return False


class KLScheduler:
    """
    Schedule linéaire pour le coefficient β du KL Annealing.

    β(t) = clip(β_start + (β_end - β_start) * t / warmup_epochs, 0, β_end)

    Parameters
    ----------
    beta_start : float
        Valeur initiale de β (généralement 0.0).
    beta_end : float
        Valeur finale de β (généralement 1.0).
    warmup_epochs : int
        Nombre d'epochs pour atteindre β_end.
    """

    def __init__(
        self,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        warmup_epochs: int = 40,
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs
        self._epoch: int = 0

    @property
    def beta(self) -> float:
        """Valeur courante de β."""
        if self.warmup_epochs == 0:
            return self.beta_end
        progress = self._epoch / self.warmup_epochs
        return float(
            np.clip(
                self.beta_start + (self.beta_end - self.beta_start) * progress,
                self.beta_start,
                self.beta_end,
            )
        )

    @property
    def warmup_done(self) -> bool:
        """True si la phase de warmup est terminée."""
        return self._epoch >= self.warmup_epochs

    def step(self) -> None:
        """Incrémente le compteur d'epoch."""
        self._epoch += 1


# ---------------------------------------------------------------------------
# Trainer principal
# ---------------------------------------------------------------------------

class VAETrainer:
    """
    Orchestrateur de l'entraînement du LSTM-VAE.

    Corrections v3 :
    - Early stopping sur val_recon (et non val_elbo)
    - Gel de l'early stopping pendant le KL warmup
    - Checkpointing sur val_recon pour cohérence avec le monitoring

    Parameters
    ----------
    cfg : VAEConfig
        Hyperparamètres du VAE et de l'entraînement.
    checkpoint_dir : Path
        Répertoire de sauvegarde des checkpoints.
    device : str, optional
        Device PyTorch ("cuda", "mps", "cpu"). Auto-détecté si None.
    """

    def __init__(
        self,
        cfg: VAEConfig,
        checkpoint_dir: Path,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        logger.info(f"VAETrainer — device : {self.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self, bundle: DataBundle
    ) -> Tuple[TimeSeriesVAE, TrainingHistory]:
        """
        Entraîne le LSTM-VAE sur le DataBundle fourni.

        Stratégie de monitoring corrigée (v3) :
        - Pendant le KL warmup : early stopping GELE, checkpoint sur val_recon
        - Après le warmup : early stopping ACTIF sur val_recon
        """
        input_dim = bundle.n_features
        seq_len = bundle.seq_len

        model = self._build_model(input_dim, seq_len)
        model = model.to(self.device)
        logger.info(f"Modèle : {model}")

        train_loader = self._make_loader(bundle.sequences_train, shuffle=True)
        val_loader = self._make_loader(bundle.sequences_val, shuffle=False)

        optimizer = AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs, eta_min=1e-6
        )

        kl_scheduler = KLScheduler(
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            warmup_epochs=self.cfg.beta_warmup_epochs,
        )

        # CORRECTION : early stopping gelé pendant le KL warmup
        early_stopping = EarlyStopping(patience=self.cfg.early_stopping_patience)
        early_stopping.freeze()
        warmup_active = True

        history = TrainingHistory()

        logger.info(
            f"Entraînement — {self.cfg.epochs} epochs max | "
            f"batch={self.cfg.batch_size} | lr={self.cfg.learning_rate} | "
            f"KL warmup={self.cfg.beta_warmup_epochs} epochs | "
            f"monitoring=val_recon [CORRIGE — pas val_elbo]"
        )

        for epoch in range(1, self.cfg.epochs + 1):
            beta = kl_scheduler.beta
            current_lr = optimizer.param_groups[0]["lr"]

            # --- Train ---
            train_metrics = self._run_epoch(
                model, train_loader, optimizer, beta, training=True
            )

            # --- Validation ---
            val_metrics = self._run_epoch(
                model, val_loader, None, beta, training=False
            )

            # --- Logging ---
            history.train_loss.append(train_metrics["elbo"])
            history.val_loss.append(val_metrics["elbo"])
            history.train_recon.append(train_metrics["recon"])
            history.val_recon.append(val_metrics["recon"])
            history.train_kl.append(train_metrics["kl"])
            history.val_kl.append(val_metrics["kl"])
            history.beta_values.append(beta)
            history.learning_rates.append(current_lr)

            # --- Dégel de l'early stopping à la fin du warmup ---
            if warmup_active and kl_scheduler.warmup_done:
                early_stopping.unfreeze()
                warmup_active = False
                # Sauvegarde forcée à la fin du warmup : c'est le premier checkpoint
                # avec β final. L'early stopping post-warmup partira de ce point.
                history.best_epoch = epoch
                self._save_checkpoint(model, optimizer, epoch, val_metrics["recon"])
                logger.info(
                    f"  Epoch {epoch} — KL warmup terminé (β={beta:.3f}). "
                    f"Early stopping activé sur val_recon. Checkpoint initial sauvegardé."
                )

            # CORRECTION : val_recon et non val_elbo
            is_best = early_stopping.step(val_metrics["recon"])
            if is_best:
                history.best_epoch = epoch
                self._save_checkpoint(model, optimizer, epoch, val_metrics["recon"])

            # --- Affichage ---
            if epoch % 10 == 0 or epoch == 1:
                warmup_tag = "warmup" if warmup_active else "actif"
                logger.info(
                    f"  Epoch {epoch:4d}/{self.cfg.epochs} | "
                    f"beta={beta:.3f} | "
                    f"train_recon={train_metrics['recon']:.5f} | "
                    f"val_recon={val_metrics['recon']:.5f} | "
                    f"val_kl={val_metrics['kl']:.5f} | "
                    f"lr={current_lr:.2e} | "
                    f"ES[{warmup_tag}] ctr={early_stopping._counter}/{self.cfg.early_stopping_patience}"
                )

            # --- KL & LR schedulers ---
            kl_scheduler.step()
            scheduler.step()

            # --- Early stopping ---
            if early_stopping.should_stop:
                logger.info(
                    f"  Early stopping à l'epoch {epoch} "
                    f"(meilleur val_recon: epoch {history.best_epoch})"
                )
                break

        # Charge le meilleur checkpoint
        model = self.load(model)
        best_recon = min(history.val_recon)
        logger.success(
            f"Entraînement terminé — meilleur val_recon={best_recon:.6f} "
            f"(epoch {history.best_epoch}/{epoch})"
        )

        history.save(self.checkpoint_dir / "training_history.json")
        return model, history

    def encode_all(
        self,
        model: TimeSeriesVAE,
        bundle: DataBundle,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extrait les représentations latentes μ pour tous les splits.
        """
        model.eval()
        model = model.to(self.device)

        def _encode_split(sequences: np.ndarray) -> np.ndarray:
            loader = self._make_loader(sequences, shuffle=False)
            latents = []
            with torch.no_grad():
                for (x_batch,) in loader:
                    x_batch = x_batch.to(self.device)
                    mu = model.encode(x_batch)
                    latents.append(mu.cpu().numpy())
            return np.concatenate(latents, axis=0)

        latent_train = _encode_split(bundle.sequences_train)
        latent_val = _encode_split(bundle.sequences_val)
        latent_test = _encode_split(bundle.sequences_test)

        logger.info(
            f"Espaces latents extraits — "
            f"train: {latent_train.shape} | "
            f"val: {latent_val.shape} | "
            f"test: {latent_test.shape}"
        )
        return latent_train, latent_val, latent_test

    def load(
        self,
        model: Optional[TimeSeriesVAE] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> TimeSeriesVAE:
        """
        Charge un modèle depuis un checkpoint.
        """
        path = checkpoint_path or (self.checkpoint_dir / "vae_best.pt")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if model is None:
            cfg = checkpoint["model_config"]
            model = TimeSeriesVAE(**cfg)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        logger.info(
            f"Checkpoint chargé : {path.name} "
            f"(epoch {checkpoint['epoch']}, val_recon={checkpoint['val_loss']:.6f})"
        )
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self, input_dim: int, seq_len: int) -> TimeSeriesVAE:
        """Construit le modèle en injectant input_dim depuis les données."""
        return TimeSeriesVAE(
            input_dim=input_dim,
            hidden_dim=self.cfg.hidden_dim,
            latent_dim=self.cfg.latent_dim,
            seq_len=seq_len,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        )

    def _make_loader(
        self, sequences: np.ndarray, shuffle: bool = False
    ) -> DataLoader:
        """Construit un DataLoader PyTorch depuis un array NumPy."""
        tensor = torch.from_numpy(sequences)
        dataset = TensorDataset(tensor)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _run_epoch(
        self,
        model: TimeSeriesVAE,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        beta: float,
        training: bool,
    ) -> Dict[str, float]:
        """
        Exécute une epoch complète (train ou validation).

        Returns
        -------
        dict
            Métriques moyennées : elbo, recon, kl.
        """
        model.train(training)
        total_elbo = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        ctx = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)

                if training:
                    optimizer.zero_grad()

                output: VAEOutput = model(x_batch, beta=beta)

                if training:
                    output.elbo.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.cfg.grad_clip
                    )
                    optimizer.step()

                total_elbo += output.elbo.item()
                total_recon += output.recon_loss.item()
                total_kl += output.kl_loss.item()
                n_batches += 1

        return {
            "elbo": total_elbo / n_batches,
            "recon": total_recon / n_batches,
            "kl": total_kl / n_batches,
        }

    def _save_checkpoint(
        self,
        model: TimeSeriesVAE,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_recon: float,
    ) -> None:
        """Sauvegarde le modèle, l'optimiseur et les métadonnées."""
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_recon,  # champ conservé pour compatibilité load()
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "latent_dim": model.latent_dim,
                "seq_len": model.seq_len,
                "num_layers": self.cfg.num_layers,
                "dropout": self.cfg.dropout,
            },
        }
        torch.save(checkpoint, self.checkpoint_dir / "vae_best.pt")