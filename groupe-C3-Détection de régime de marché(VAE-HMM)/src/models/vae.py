"""
models/vae.py
=============
Variational Autoencoder adapté aux séries temporelles financières.

Architecture : LSTM-VAE avec Temporal Attention Pooling (v2)

Corrections apportées :
  - Bug #3 : Encodeur bidirectionnel corrigé.
    L'ancien code prenait lstm_out[:, -1, :] pour les deux directions :
      - forward :  lstm_out[:, -1, :hidden_dim]  ← h_{T} forward  ✓ (voit tout le passé)
      - backward : lstm_out[:, -1, hidden_dim:]  ← h_{T} backward ✗ (voit seulement z_T)
    Correction : lstm_out[:, 0, hidden_dim:] pour la direction backward (h_0^{bwd})
    qui a vu toute la séquence de T à 0.

  - Amélioration : Temporal Attention Pooling remplace le last-step.
    Un mécanisme d'attention apprend quels pas de temps sont les plus informatifs
    pour caractériser le régime, au lieu de n'utiliser qu'un seul timestep.

Classes
-------
TemporalAttention
    Attention additive sur les sorties LSTM — apprend l'importance de chaque t.
LSTMEncoder
    Encode (T, F) → (μ, log σ²) via LSTM bidir + attention pooling corrigé.
LSTMDecoder
    Décode z → (T, F).
TimeSeriesVAE
    Modèle complet : Encoder + Decoder + ELBO loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class VAEOutput:
    """Conteneur des sorties du forward pass du VAE."""
    x_recon: Tensor
    mu: Tensor
    log_var: Tensor
    z: Tensor
    recon_loss: Tensor
    kl_loss: Tensor
    elbo: Tensor


# ---------------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Mécanisme d'attention additive (Bahdanau) sur les sorties LSTM.

    Apprend un score d'importance s_t = v^T tanh(W h_t + b) pour chaque
    pas de temps t, puis calcule le contexte agrégé comme la somme pondérée :
      context = Σ_t softmax(s_t) × h_t

    Avantage vs last-step : le modèle peut apprendre à se concentrer sur
    les periodes de stress, les pics de volatilité, ou tout autre pattern
    temporel informatif pour identifier le régime.

    Parameters
    ----------
    hidden_dim : int
        Dimension des états cachés LSTM (après concaténation bidir = 2×hidden_dim).
    attention_dim : int
        Dimension de l'espace d'attention interne. Par défaut = hidden_dim // 2.
    """

    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None) -> None:
        super().__init__()
        attention_dim = attention_dim or max(hidden_dim // 2, 16)

        # Projection vers l'espace d'attention
        self.W = nn.Linear(hidden_dim, attention_dim, bias=True)
        # Vecteur de score (dot product)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        # Initialisation Xavier
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, lstm_out: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calcule le contexte agrégé par attention.

        Parameters
        ----------
        lstm_out : Tensor
            Sorties LSTM — shape (B, T, hidden_dim).

        Returns
        -------
        context : Tensor
            Vecteur agrégé — shape (B, hidden_dim).
        attention_weights : Tensor
            Poids d'attention normalisés — shape (B, T).
            Utile pour la visualisation des régimes.
        """
        # Scores d'attention : (B, T, attention_dim) → (B, T, 1)
        energy = torch.tanh(self.W(lstm_out))   # (B, T, attention_dim)
        scores = self.v(energy).squeeze(-1)      # (B, T)

        # Normalisation softmax sur la dimension temporelle
        attention_weights = F.softmax(scores, dim=-1)  # (B, T)

        # Contexte agrégé : somme pondérée des états cachés
        context = torch.bmm(
            attention_weights.unsqueeze(1),   # (B, 1, T)
            lstm_out,                         # (B, T, hidden_dim)
        ).squeeze(1)                          # (B, hidden_dim)

        return context, attention_weights


# ---------------------------------------------------------------------------
# Encoder — CORRIGÉ + Attention Pooling
# ---------------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    """
    Encodeur LSTM bidirectionnel avec Temporal Attention Pooling.

    Architecture corrigée :
      X → [LSTM bidir, L couches] → Attention Pooling → LayerNorm → (μ, log σ²)

    Correction du bug bidirectionnel :
      Ancien : h_last = lstm_out[:, -1, :]
        → direction backward ignorée (lstm_out[:, -1, hidden_dim:] = h_T backward
           qui n'a vu qu'un seul pas de temps en direction inverse)
      Nouveau : attention pooling sur toutes les sorties (B, T, 2*H)
        → chaque direction contribue à toutes les positions temporelles

    Parameters
    ----------
    input_dim : int
        Dimension des features d'entrée F.
    hidden_dim : int
        Dimension de l'état caché LSTM par direction.
    latent_dim : int
        Dimension de l'espace latent.
    num_layers : int
        Couches LSTM empilées.
    dropout : float
        Taux de dropout.
    use_attention : bool
        Si True (défaut), utilise Temporal Attention Pooling.
        Si False, utilise la concaténation forward[-1] + backward[0] (plus simple).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if use_attention:
            # Attention sur les sorties concaténées (2*hidden_dim)
            self.attention = TemporalAttention(2 * hidden_dim)
        else:
            # Fallback : pas d'attention
            self.attention = None

        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc_mu = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(2 * hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialisation orthogonale des poids LSTM."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass de l'encodeur avec pooling corrigé.

        Parameters
        ----------
        x : Tensor
            Séquence d'entrée — shape (B, T, F).

        Returns
        -------
        mu : Tensor — shape (B, latent_dim)
        log_var : Tensor — shape (B, latent_dim)
        """
        lstm_out, _ = self.lstm(x)   # (B, T, 2*hidden_dim)

        if self.use_attention:
            # Temporal Attention Pooling : apprend quels t sont importants
            h_pooled, _ = self.attention(lstm_out)   # (B, 2*hidden_dim)
        else:
            # Correction bug bidirectionnel : forward[-1] + backward[0]
            # Ancien bug : lstm_out[:, -1, :] prenait backward à t=T
            # (backward h_{T} a vu seulement z_{T}, pas toute la séquence)
            h_forward = lstm_out[:, -1, :self.hidden_dim]    # h_T forward ✓
            h_backward = lstm_out[:, 0, self.hidden_dim:]    # h_0 backward ✓
            h_pooled = torch.cat([h_forward, h_backward], dim=-1)

        h_pooled = self.layer_norm(h_pooled)
        h_pooled = self.dropout(h_pooled)

        mu = self.fc_mu(h_pooled)
        log_var = self.fc_log_var(h_pooled)
        log_var = log_var.clamp(-10.0, 4.0)

        return mu, log_var


# ---------------------------------------------------------------------------
# Decoder (inchangé)
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """
    Décodeur LSTM pour la reconstruction de séquences.

    Architecture :
      z → [Linear → Tanh] → h_0, c_0 → [LSTM, L couches] → [Linear] → X̂
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.z_to_hidden = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.z_to_cell = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.z_to_input = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor) -> Tensor:
        """Reconstruit une séquence à partir du vecteur latent."""
        batch_size = z.size(0)

        h0 = torch.tanh(self.z_to_hidden(z))
        c0 = torch.tanh(self.z_to_cell(z))

        h0 = h0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = c0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        dec_input = torch.tanh(self.z_to_input(z))
        dec_input = dec_input.unsqueeze(1).repeat(1, self.seq_len, 1)

        lstm_out, _ = self.lstm(dec_input, (h0, c0))
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        return self.output_projection(lstm_out)


# ---------------------------------------------------------------------------
# VAE complet
# ---------------------------------------------------------------------------

class TimeSeriesVAE(nn.Module):
    """
    LSTM-VAE avec Temporal Attention Pooling pour séries temporelles financières.

    Améliorations v2 :
    - Encodeur bidirectionnel corrigé (bug forward/backward pooling)
    - Temporal Attention Pooling : le modèle apprend quels pas de temps
      sont les plus informatifs pour l'identification du régime
    - Exposition des poids d'attention pour la visualisation/interprétabilité
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
        )
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
        )

        self._n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def n_params(self) -> int:
        return self._n_params

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick (Kingma & Welling, 2013)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x: Tensor, beta: float = 1.0) -> VAEOutput:
        """Forward pass avec calcul de la loss ELBO."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)

        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        elbo = recon_loss + beta * kl_loss

        return VAEOutput(
            x_recon=x_recon,
            mu=mu,
            log_var=log_var,
            z=z,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            elbo=elbo,
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encode et retourne μ (déterministe, pour l'inférence)."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

    def get_attention_weights(self, x: Tensor) -> Optional[Tensor]:
        """
        Retourne les poids d'attention temporelle pour une séquence.

        Utile pour visualiser quels pas de temps ont influencé
        l'identification du régime.

        Returns
        -------
        Tensor ou None
            Poids d'attention — shape (B, T), ou None si pas d'attention.
        """
        if not hasattr(self.encoder, 'attention') or self.encoder.attention is None:
            return None
        self.eval()
        with torch.no_grad():
            lstm_out, _ = self.encoder.lstm(x)
            _, attn_weights = self.encoder.attention(lstm_out)
        return attn_weights

    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstruit une séquence."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x, beta=1.0)
        return out.x_recon

    def sample(self, n_samples: int, device: torch.device) -> Tensor:
        """Génère des séquences depuis le prior p(z) = N(0, I)."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            return self.decoder(z)

    def __repr__(self) -> str:
        return (
            f"TimeSeriesVAE("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"seq_len={self.seq_len}, "
            f"attention=True, "
            f"params={self.n_params:,})"
        )