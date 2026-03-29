"""
gnn_model.py
============
Implémentation des modèles GCN et GAT pour la prédiction de rendements.

Deux modèles :
  - GCNModel : Graph Convolutional Network (Kipf & Welling 2017) — niveau minimum
  - GATModel : Graph Attention Network (Veličković et al. 2018)  — niveau bon/excellent

Usage:
    from src.gnn_model import GCNModel, GATModel, GNNTrainer
    model   = GATModel(in_channels=5, hidden=64, out_channels=1, heads=4)
    trainer = GNNTrainer(model, lr=1e-3)
    trainer.fit(train_graphs, train_targets, epochs=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


# ─────────────────────────────────────────────────────────────────────────────
# 1. GCN — Graph Convolutional Network
# ─────────────────────────────────────────────────────────────────────────────
class GCNModel(nn.Module):
    """
    Graph Convolutional Network pour la prédiction de rendements.

    Architecture : Input → GCNConv → ReLU → Dropout → GCNConv → Linear

    Principe (Kipf & Welling 2017)
    ------------------------------
    Chaque couche GCN applique :
        H^(l+1) = σ( D^{-1/2} Â D^{-1/2} H^(l) W^(l) )
    où :
        Â = A + I   (matrice d'adjacence + auto-boucles)
        D = degrés  (normalisation)
        W = paramètres apprenables
        σ = fonction d'activation (ReLU)

    En termes simples : chaque nœud agrège la moyenne pondérée de ses voisins.

    Paramètres
    ----------
    in_channels  : int — nombre de features par nœud en entrée
    hidden       : int — taille des couches cachées (défaut 64)
    out_channels : int — nombre de sorties par nœud (1 = rendement scalaire)
    dropout      : float — taux de dropout pour régularisation
    """

    def __init__(
        self,
        in_channels  : int,
        hidden       : int   = 64,
        out_channels : int   = 1,
        dropout      : float = 0.3,
    ):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden)
        self.conv2   = GCNConv(hidden, hidden // 2)
        self.head    = nn.Linear(hidden // 2, out_channels)
        self.dropout = dropout

        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier pour stabiliser l'entraînement."""
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Passage avant.

        Paramètres
        ----------
        x          : Tensor (n_noeuds × in_channels)
        edge_index : Tensor (2 × n_aretes)
        edge_attr  : Tensor (n_aretes × 1) optionnel — poids des arêtes

        Retourne
        --------
        out : Tensor (n_noeuds × out_channels) — rendements prédits
        """
        # Couche 1 : message passing + activation
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Couche 2 : deuxième agrégation
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Tête de prédiction linéaire
        out = self.head(h)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAT — Graph Attention Network
# ─────────────────────────────────────────────────────────────────────────────
class GATModel(nn.Module):
    """
    Graph Attention Network pour la prédiction de rendements.

    Architecture : Input → GATConv (multi-head) → ELU → Dropout → GATConv → Linear

    Principe (Veličković et al. 2018)
    ----------------------------------
    Contrairement au GCN qui agrège uniformément les voisins, le GAT
    apprend un coefficient d'attention α_{ij} pour chaque arête :

        α_{ij} = softmax( LeakyReLU( a^T [W·h_i || W·h_j] ) )
        h'_i   = σ( Σ_{j∈N(i)} α_{ij} · W·h_j )

    Avantages par rapport au GCN :
      - Pondère différemment chaque voisin selon son "importance"
      - Multi-head attention : plusieurs perspectives simultanées
      - Plus adaptatif aux changements de structure du marché

    Référence : Zhang (2026) Fig. 2 — Graph Attention Layer

    Paramètres
    ----------
    in_channels  : int — features en entrée
    hidden       : int — dimension cachée par tête
    out_channels : int — sorties par nœud
    heads        : int — nombre de têtes d'attention (défaut 4)
    dropout      : float — dropout
    """

    def __init__(
        self,
        in_channels  : int,
        hidden       : int   = 64,
        out_channels : int   = 1,
        heads        : int   = 4,
        dropout      : float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Couche 1 : multi-head attention (concat les têtes → hidden * heads)
        self.gat1 = GATConv(
            in_channels,
            hidden,
            heads       = heads,
            dropout     = dropout,
            concat      = True,    # concatène les h têtes
        )

        # Couche 2 : attention mono-tête (moyenne)
        self.gat2 = GATConv(
            hidden * heads,
            hidden,
            heads       = 1,
            dropout     = dropout,
            concat      = False,   # moyenne des têtes
        )

        # Couche de normalisation (stabilise l'entraînement)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)

        # Tête de prédiction
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_channels),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Paramètres
        ----------
        x          : Tensor (n_noeuds × in_channels)
        edge_index : Tensor (2 × n_aretes)
        edge_attr  : ignoré pour GATConv standard

        Retourne
        --------
        out             : Tensor (n_noeuds × out_channels)
        attention_weights : list of Tensor — poids d'attention par couche
        """
        # Couche 1 avec attention
        h, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Couche 2 avec attention
        h, attn2 = self.gat2(h, edge_index, return_attention_weights=True)
        h = self.norm2(h)
        h = F.elu(h)

        out = self.head(h)
        return out, [attn1, attn2]

    def forward_simple(self, x, edge_index, edge_attr=None):
        """Version simplifiée sans retour des poids d'attention (pour l'entraînement)."""
        out, _ = self.forward(x, edge_index, edge_attr)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Préparation des cibles (rendements futurs)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_targets(
    returns       : 'pd.DataFrame',
    horizon       : int   = 5,
    normalize     : bool  = True,
) -> tuple:
    """
    Prépare les cibles d'entraînement : rendements futurs sur `horizon` jours.

    Paramètres
    ----------
    returns  : DataFrame (n_jours × n_actifs)
    horizon  : int — nombre de jours à prédire (défaut 5 = 1 semaine)
    normalize: bool — normaliser les cibles (z-score)

    Retourne
    --------
    (X_dates, y) : (index des dates valides, ndarray n_dates × n_actifs)

    Note pédagogique
    ----------------
    On prédit le rendement cumulé sur les `horizon` prochains jours :
        y_t = Σ_{k=1}^{horizon} r_{t+k}
    C'est ce que le portefeuille va capturer si on rééquilibre toutes les semaines.
    """
    # Rendement cumulé sur horizon jours (rolling sum vers l'avant)
    y = returns.shift(-horizon).rolling(horizon).sum()

    # Supprimer les lignes avec NaN (début et fin)
    valid_mask = ~y.isna().any(axis=1)
    y_clean    = y[valid_mask]
    x_dates    = returns.index[valid_mask]

    y_values = y_clean.values.astype(np.float32)

    if normalize:
        # Z-score par actif (stabilise l'entraînement)
        mean = y_values.mean(axis=0, keepdims=True)
        std  = y_values.std(axis=0, keepdims=True) + 1e-8
        y_values = (y_values - mean) / std

    return x_dates, y_values


# ─────────────────────────────────────────────────────────────────────────────
# 4. Entraîneur (Trainer)
# ─────────────────────────────────────────────────────────────────────────────
class GNNTrainer:
    """
    Gère l'entraînement, la validation et l'évaluation du GNN.

    Paramètres
    ----------
    model   : GCNModel ou GATModel
    lr      : float — learning rate (défaut 1e-3)
    weight_decay : float — L2 régularisation (défaut 1e-4)
    device  : 'cuda' ou 'cpu' (auto-détecté)
    """

    def __init__(
        self,
        model        : nn.Module,
        lr           : float = 1e-3,
        weight_decay : float = 1e-4,
        device       : str   = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = model.to(self.device)
        self.opt    = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=10, factor=0.5
        )

        self.train_losses = []
        self.val_losses   = []

        print(f"[GNNTrainer] Device : {self.device}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[GNNTrainer] Paramètres : {n_params:,}")

    def _forward(self, data):
        """Appel unifié pour GCN et GAT."""
        x   = data.x.to(self.device)
        ei  = data.edge_index.to(self.device)
        ea  = data.edge_attr.to(self.device) if data.edge_attr is not None else None

        if isinstance(self.model, GATModel):
            out = self.model.forward_simple(x, ei, ea)
        else:
            out = self.model(x, ei, ea)

        return out  # (n_noeuds × out_channels)

    def fit(
        self,
        graph_data    : 'Data',
        targets       : np.ndarray,
        epochs        : int   = 200,
        val_split     : float = 0.2,
        verbose_every : int   = 20,
    ):
        """
        Entraîne le modèle.

        Note sur la stratégie d'entraînement
        -------------------------------------
        On utilise un seul graphe (statique) et on fait varier les
        features des nœuds + les cibles dans le temps.
        La validation utilise les derniers `val_split`% des timesteps
        pour éviter le data leakage (jamais de mélange aléatoire en finance !).

        Paramètres
        ----------
        graph_data : Data PyG — graphe (edge_index fixe pour graphe statique)
        targets    : ndarray (n_timesteps × n_actifs) — rendements futurs
        epochs     : int
        val_split  : float — fraction pour la validation (fin de la série)
        """
        n          = len(targets)
        val_start  = int(n * (1 - val_split))
        train_targets = torch.tensor(targets[:val_start],  dtype=torch.float)
        val_targets   = torch.tensor(targets[val_start:],  dtype=torch.float)

        print(f"[GNNTrainer] Train: {val_start} timesteps | "
              f"Val: {n - val_start} timesteps")

        best_val   = float('inf')
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()

            # Un seul pass sur le graphe avec features moyennées sur la période d'entraînement
            self.opt.zero_grad()
            pred = self._forward(graph_data)   # (n_actifs × 1)

            # Cible : rendement moyen sur la période d'entraînement
            target_mean = train_targets.mean(dim=0).unsqueeze(1).to(self.device)
            loss = F.mse_loss(pred, target_mean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                pred_val     = self._forward(graph_data)
                target_val   = val_targets.mean(dim=0).unsqueeze(1).to(self.device)
                val_loss     = F.mse_loss(pred_val, target_val).item()

            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % verbose_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{epochs} | "
                      f"Train loss: {loss.item():.6f} | "
                      f"Val loss: {val_loss:.6f}")

        # Restaurer le meilleur modèle
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"\n[GNNTrainer] ✓ Entraînement terminé | "
              f"Meilleure val loss: {best_val:.6f}")

    def predict(self, graph_data: 'Data') -> np.ndarray:
        """
        Prédit les rendements futurs pour chaque actif.

        Retourne
        --------
        preds : ndarray (n_actifs,) — scores de rendement prédit
        """
        self.model.eval()
        with torch.no_grad():
            out = self._forward(graph_data)
        return out.squeeze(-1).cpu().numpy()

    def plot_training(self, figsize=(10, 4)):
        """Courbes d'entraînement : train loss vs validation loss."""
        fig, ax = plt.subplots(figsize=figsize)
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, label='Train loss',
                color='#3266ad', linewidth=1.5)
        ax.plot(epochs, self.val_losses,   label='Val loss',
                color='#E85D24', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Époque')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Courbes de perte — GNN', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def save(self, path: str):
        """Sauvegarde les poids du modèle."""
        torch.save({
            'model_state': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses':   self.val_losses,
        }, path)
        print(f"[GNNTrainer] Modèle sauvegardé : {path}")

    def load(self, path: str):
        """Charge les poids du modèle."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses   = ckpt.get('val_losses',   [])
        print(f"[GNNTrainer] Modèle chargé : {path}")
