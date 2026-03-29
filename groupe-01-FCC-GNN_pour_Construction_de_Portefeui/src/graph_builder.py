"""
graph_builder.py
================
Construit le graphe financier entre actifs à partir de la matrice de corrélation.

Deux modes :
  - Graphe statique  : calculé sur toute la période
  - Graphe dynamique : rolling window (pour le niveau excellent)

Produit un objet torch_geometric.data.Data prêt à être consommé par le GNN.

Usage:
    from src.graph_builder import GraphBuilder
    builder = GraphBuilder(returns, threshold=0.3)
    data    = builder.build_static_graph(node_features)
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')


class GraphBuilder:
    """
    Transforme une matrice de corrélation en graphe pour le GNN.

    Paramètres
    ----------
    returns   : pd.DataFrame (n_jours × n_actifs) — rendements log
    threshold : float — seuil minimal |ρ| pour créer une arête (défaut 0.3)
    sectors   : dict optionnel — ticker → secteur (pour colorer le graphe)

    Concept
    -------
    Un graphe G = (V, E) où :
      - V = ensemble des actifs (nœuds)
      - E = paires (i, j) telles que |ρ(i,j)| > threshold
      - w(i,j) = |ρ(i,j)| — poids de l'arête
    """

    def __init__(
        self,
        returns   : pd.DataFrame,
        threshold : float = 0.3,
        sectors   : dict  = None,
    ):
        self.returns   = returns
        self.threshold = threshold
        self.sectors   = sectors or {}
        self.tickers   = list(returns.columns)
        self.n         = len(self.tickers)

        # Calculés à la demande
        self._corr_matrix = None
        self._nx_graph    = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Matrice de corrélation
    # ─────────────────────────────────────────────────────────────────────────
    def compute_correlation(self, returns: pd.DataFrame = None) -> np.ndarray:
        """
        Calcule la matrice de corrélation de Pearson.

        Paramètres
        ----------
        returns : DataFrame optionnel — si None, utilise self.returns

        Retourne
        --------
        corr : ndarray (n × n)
        """
        r = returns if returns is not None else self.returns
        corr = r.corr().values.astype(np.float32)
        # Mettre la diagonale à 0 (pas d'auto-corrélation comme arête)
        np.fill_diagonal(corr, 0.0)
        if returns is None:
            self._corr_matrix = corr
        return corr

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Graphe statique → objet PyTorch Geometric
    # ─────────────────────────────────────────────────────────────────────────
    def build_static_graph(self, node_features: np.ndarray) -> Data:
        """
        Construit le graphe financier statique (calculé sur toute la période).

        Paramètres
        ----------
        node_features : ndarray (n_actifs × n_features)
            Features initiales de chaque nœud (sortie de DataLoader.build_node_features)

        Retourne
        --------
        data : torch_geometric.data.Data
            Objet standard PyG contenant :
              - data.x          : features des nœuds  (n × F)
              - data.edge_index : liste des arêtes     (2 × E)
              - data.edge_attr  : poids des arêtes     (E × 1)
              - data.num_nodes  : nombre de nœuds

        Note sur edge_index
        -------------------
        PyTorch Geometric représente le graphe par une matrice (2 × E) :
          edge_index[0] = nœuds sources
          edge_index[1] = nœuds destinations
        Le graphe est non-orienté → chaque arête (i,j) apparaît deux fois.
        """
        corr = self.compute_correlation()

        # Extraire les arêtes dont |ρ| > threshold
        rows, cols = np.where(np.abs(corr) > self.threshold)

        # Supprimer la diagonale (auto-boucles)
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]

        edge_weights = np.abs(corr[rows, cols])

        # Conversion en tenseurs PyTorch
        edge_index = torch.tensor(
            np.stack([rows, cols], axis=0),
            dtype=torch.long
        )
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        x         = torch.tensor(node_features,  dtype=torch.float)

        data = Data(
            x          = x,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            num_nodes  = self.n,
        )

        print(f"[GraphBuilder] Graphe statique :")
        print(f"  Nœuds   : {data.num_nodes}")
        print(f"  Arêtes  : {edge_index.shape[1] // 2} paires "
              f"(threshold={self.threshold})")
        print(f"  Densité : {edge_index.shape[1] / (self.n*(self.n-1)):.1%}")
        print(f"  Features: {x.shape[1]} par nœud")

        # Construire aussi le graphe networkx pour visualisation
        self._build_nx_graph(corr)

        return data

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Graphe dynamique — rolling window (niveau excellent)
    # ─────────────────────────────────────────────────────────────────────────
    def build_dynamic_graphs(
        self,
        node_features_fn,
        window    : int = 60,
        step      : int = 20,
    ) -> list:
        """
        Construit une séquence de graphes sur des fenêtres glissantes.

        Paramètres
        ----------
        node_features_fn : callable(returns_window) → ndarray (n × F)
            Fonction qui calcule les features des nœuds pour une fenêtre donnée.
        window : int — taille de la fenêtre en jours de trading (défaut 60 ≈ 3 mois)
        step   : int — pas entre deux fenêtres (défaut 20 ≈ 1 mois)

        Retourne
        --------
        graphs : list of (date, Data)
            Liste de tuples (date de fin de fenêtre, graphe PyG)

        Pourquoi les graphes dynamiques ?
        ----------------------------------
        La corrélation entre actifs n'est pas stable dans le temps.
        Pendant une crise (ex: COVID mars 2020), TOUTES les corrélations
        tendent vers 1 (contagion). En période normale, les clusters
        sectoriels sont plus distincts.
        Le graphe dynamique permet au GNN de s'adapter à ces régimes.
        """
        graphs = []
        dates  = self.returns.index

        n_windows = (len(dates) - window) // step + 1
        print(f"[GraphBuilder] Graphes dynamiques : "
              f"{n_windows} fenêtres × {window}j (step={step}j)")

        for start_idx in range(0, len(dates) - window, step):
            end_idx = start_idx + window
            window_returns = self.returns.iloc[start_idx:end_idx]
            end_date       = dates[end_idx - 1]

            # Corrélation sur cette fenêtre
            corr = self.compute_correlation(window_returns)

            # Features des nœuds sur cette fenêtre
            node_feat = node_features_fn(window_returns)

            # Construire le graphe
            rows, cols = np.where(np.abs(corr) > self.threshold)
            mask = rows != cols
            rows, cols = rows[mask], cols[mask]

            if len(rows) == 0:
                # Fenêtre sans arêtes (rare) : graphe vide
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr  = torch.zeros((0, 1), dtype=torch.float)
            else:
                edge_weights = np.abs(corr[rows, cols])
                edge_index = torch.tensor(
                    np.stack([rows, cols], axis=0), dtype=torch.long
                )
                edge_attr = torch.tensor(
                    edge_weights, dtype=torch.float
                ).unsqueeze(1)

            x    = torch.tensor(node_feat, dtype=torch.float)
            data = Data(
                x          = x,
                edge_index = edge_index,
                edge_attr  = edge_attr,
                num_nodes  = self.n,
            )
            graphs.append((end_date, data))

        print(f"[GraphBuilder] ✓ {len(graphs)} graphes construits")
        return graphs

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Graphe NetworkX (pour visualisation)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_nx_graph(self, corr: np.ndarray):
        """Construit le graphe networkx pour visualisation."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(corr[i, j]) > self.threshold:
                    G.add_edge(i, j, weight=abs(corr[i, j]),
                               sign=np.sign(corr[i, j]))

        nx.set_node_attributes(
            G,
            {i: self.tickers[i] for i in range(self.n)},
            'ticker'
        )
        nx.set_node_attributes(
            G,
            {i: self.sectors.get(self.tickers[i], 'Unknown')
             for i in range(self.n)},
            'sector'
        )
        self._nx_graph = G
        return G

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Visualisations
    # ─────────────────────────────────────────────────────────────────────────
    def plot_graph(self, figsize=(14, 10), title=None):
        """
        Visualise le graphe financier avec networkx.
        - Nœuds colorés par secteur
        - Épaisseur des arêtes proportionnelle à |ρ|
        - Taille des nœuds proportionnelle au degré (connectivité)
        """
        if self._nx_graph is None:
            raise RuntimeError("Appelez d'abord build_static_graph().")

        G = self._nx_graph

        sector_palette = {
            'Technology':    '#3266ad',
            'Financials':    '#E85D24',
            'Healthcare':    '#1D9E75',
            'Consumer Disc': '#EF9F27',
            'Consumer Stap': '#97C459',
            'Industrials':   '#7F77DD',
            'Energy':        '#D4537E',
            'Materials':     '#888780',
            'Telecom':       '#B0B0A0',
            'Unknown':       '#cccccc',
        }

        fig, ax = plt.subplots(figsize=figsize)

        # Layout : Kamada-Kawai (respecte les poids → actifs corrélés proches)
        pos = nx.kamada_kawai_layout(G, weight='weight')

        # Couleurs et tailles des nœuds
        node_colors = [
            sector_palette.get(
                G.nodes[i].get('sector', 'Unknown'), '#cccccc'
            )
            for i in G.nodes
        ]
        degrees    = dict(G.degree())
        node_sizes = [300 + degrees[i] * 80 for i in G.nodes]

        # Épaisseur des arêtes
        edges      = list(G.edges(data=True))
        edge_widths = [d['weight'] * 4 for _, _, d in edges]
        edge_colors = ['#3266ad' if d.get('sign', 1) > 0 else '#E85D24'
                       for _, _, d in edges]

        # Dessin
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths,
                               edge_color=edge_colors, alpha=0.5, ax=ax)
        labels = {i: G.nodes[i]['ticker'] for i in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=9,
                                font_weight='bold', ax=ax)

        # Légende secteurs
        from matplotlib.patches import Patch
        present = {G.nodes[i].get('sector', 'Unknown') for i in G.nodes}
        legend_handles = [
            Patch(color=c, label=s)
            for s, c in sector_palette.items()
            if s in present
        ]
        ax.legend(handles=legend_handles, loc='lower left',
                  fontsize=9, title='Secteur', title_fontsize=10)

        t = title or (f"Graphe financier DJIA — seuil |ρ| > {self.threshold}\n"
                      f"{G.number_of_nodes()} nœuds · "
                      f"{G.number_of_edges()} arêtes · "
                      f"Épaisseur ∝ corrélation · "
                      f"Taille ∝ connectivité")
        ax.set_title(t, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig

    def plot_correlation_threshold_analysis(self, figsize=(10, 4)):
        """
        Affiche le nombre d'arêtes et la densité du graphe en fonction du seuil.
        Aide à choisir le bon threshold.
        """
        if self._corr_matrix is None:
            self.compute_correlation()

        corr = self._corr_matrix
        upper = corr[np.triu_indices_from(corr, k=1)]

        thresholds = np.linspace(0.0, 0.9, 50)
        n_edges    = [np.sum(np.abs(upper) > t) for t in thresholds]
        densities  = [e / (self.n * (self.n - 1) / 2) for e in n_edges]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(thresholds, n_edges, color='#3266ad', linewidth=2)
        ax1.axvline(self.threshold, color='#E85D24', linestyle='--',
                    label=f'Seuil actuel = {self.threshold}')
        ax1.set_xlabel('Seuil |ρ|')
        ax1.set_ylabel("Nombre d'arêtes")
        ax1.set_title("Arêtes vs seuil")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(thresholds, [d * 100 for d in densities],
                 color='#1D9E75', linewidth=2)
        ax2.axvline(self.threshold, color='#E85D24', linestyle='--',
                    label=f'Seuil actuel = {self.threshold}')
        ax2.set_xlabel('Seuil |ρ|')
        ax2.set_ylabel('Densité (%)')
        ax2.set_title('Densité du graphe vs seuil')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Choix du seuil de corrélation pour le graphe',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_dynamic_graph_evolution(
        self,
        graphs : list,
        figsize=(14, 4),
    ):
        """
        Visualise l'évolution du nombre d'arêtes et de la densité du graphe
        au fil du temps (pour les graphes dynamiques).
        """
        dates    = [d for d, _ in graphs]
        n_edges  = [g.edge_index.shape[1] // 2 for _, g in graphs]
        avg_corr = []

        for _, g in graphs:
            if g.edge_index.shape[1] > 0:
                avg_corr.append(g.edge_attr.mean().item())
            else:
                avg_corr.append(0.0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(dates, n_edges, color='#3266ad', linewidth=1.5)
        ax1.fill_between(dates, n_edges, alpha=0.2, color='#3266ad')
        ax1.set_title("Nombre d'arêtes dans le temps")
        ax1.set_ylabel("Arêtes")
        ax1.grid(True, alpha=0.3)

        ax2.plot(dates, avg_corr, color='#E85D24', linewidth=1.5)
        ax2.fill_between(dates, avg_corr, alpha=0.2, color='#E85D24')
        ax2.set_title("Corrélation moyenne dans le temps")
        ax2.set_ylabel("|ρ| moyen")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            "Évolution du graphe dynamique (rolling correlations)\n"
            "Pics = périodes de stress marché (COVID, inflation 2022)",
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()
        return fig
