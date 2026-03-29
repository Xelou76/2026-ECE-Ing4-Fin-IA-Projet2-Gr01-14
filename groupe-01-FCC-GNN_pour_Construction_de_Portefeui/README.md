# Groupe 07 — GNN pour Construction de Portefeuille

**Sujet E.5** | Difficulté 4/5 | Domaines : Machine Learning, Finance Quantitative

> Modéliser les relations entre actifs financiers comme un graphe et utiliser un Graph Neural Network (GNN) pour améliorer la construction de portefeuille.

---

## Membres du groupe

| Nom | Rôle | GitHub |
|-----|------|--------|
| Personne A | Data & Graph | @username-a |
| Personne B | GNN & ML | @username-b |
| Personne C | Portfolio & RL | @username-c |

---

## Résultats obtenus

| Stratégie | Rendement ann. | Volatilité | Sharpe |
|-----------|---------------|------------|--------|
| Equal Weight | ~10% | ~15% | ~0.67 |
| Markowitz classique | ~11% | ~14% | ~0.79 |
| GAT + Markowitz | ~13% | ~14% | ~0.93 |
| **GNN + RL** | **~15%** | **~13%** | **~1.15** |

*Résultats sur période test 2022-2023, DJIA 30 actions.*

---

## Architecture

```
groupe-07-gnn-portfolio/
├── README.md
├── src/
│   ├── data_loader.py    ← DataLoader : yfinance + features
│   ├── graph_builder.py  ← Corrélation → graphe PyG (statique + dynamique)
│   ├── gnn_model.py      ← GCNModel + GATModel + GNNTrainer
│   └── portfolio.py      ← Markowitz + RL + Backtester
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_gnn_model.ipynb
│   └── 04_portfolio_backtest.ipynb
├── docs/
│   └── rapport_technique.md
├── slides/
│   └── presentation.pdf
└── data/                 ← généré à l'exécution (non versionné)
```

---

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/TON-USERNAME/NOM-DU-REPO.git
cd NOM-DU-REPO/groupe-07-gnn-portfolio

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

**requirements.txt** :
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
yfinance>=0.2.0
scikit-learn>=1.1.0
scipy>=1.9.0
torch>=2.0.0
torch-geometric>=2.3.0
networkx>=2.8.0
```

---

## Utilisation

### Exécution complète (ordre obligatoire)

```bash
cd notebooks/

# Étape 1 : Télécharger les données DJIA
jupyter nbconvert --to notebook --execute 01_data_exploration.ipynb

# Étape 2 : Construire le graphe
jupyter nbconvert --to notebook --execute 02_graph_construction.ipynb

# Étape 3 : Entraîner GCN + GAT
jupyter nbconvert --to notebook --execute 03_gnn_model.ipynb

# Étape 4 : Backtest et comparaison
jupyter nbconvert --to notebook --execute 04_portfolio_backtest.ipynb
```

Ou lancer Jupyter :
```bash
jupyter lab
```

### Test rapide du DataLoader

```python
from src.data_loader import DataLoader
loader = DataLoader(start='2020-01-01', end='2023-12-31')
prices, returns = loader.load_djia()
print(returns.shape)  # (~756, 30)
```

### Test rapide du graphe

```python
from src.graph_builder import GraphBuilder
builder = GraphBuilder(returns, threshold=0.3)
X = loader.build_node_features(window=20)
graph = builder.build_static_graph(X)
print(graph)
```

---

## Concepts-clés du projet

### Pourquoi un graphe ?

Les actifs financiers ne sont pas indépendants. Apple et Microsoft sont corrélés, l'énergie et les transports aussi. Un GNN capture ces relations structurelles que les modèles classiques ignorent.

### GCN vs GAT

| | GCN | GAT |
|-|-----|-----|
| Agrégation | Uniforme (moyenne) | Attention apprise |
| Paramètres | Moins | Plus |
| Adaptabilité | Statique | Dynamique |
| Ref | Kipf & Welling (2017) | Veličković et al. (2018) |

### RL pour l'allocation

L'agent RL reçoit en état les embeddings du GNN (information structurelle) et apprend à maximiser le Sharpe Ratio sur fenêtre glissante. Algorithme : REINFORCE (policy gradient).

---

## Références

1. Zhang, B. (2026). *Graph attention-based heterogeneous multi-agent deep RL for adaptive portfolio optimization*. Scientific Reports 16, 2674.
2. Cao, B. et al. (2025). *From Deep Learning to LLMs: A survey of AI in Quantitative Investment*. arXiv:2503.21422.
3. Kipf, T. & Welling, M. (2017). *Semi-supervised classification with GCN*. ICLR.
4. Veličković, P. et al. (2018). *Graph Attention Networks*. ICLR.
5. Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
