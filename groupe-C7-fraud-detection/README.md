# C.7 - Détection de Fraude en Temps Réel

**ECE Paris - Ing4 Finance - IA Probabiliste, Théorie des Jeux et ML**  
**Groupe C7** | Tom Beckermann, [Membre 2], [Membre 3]  
**GitHub** : [@Tombeck7], [@username2], [@username3]

---

## Description

Détection de fraude bancaire sur le dataset Kaggle Credit Card Fraud (284 807 transactions, 0.17% de fraudes).  
Comparaison de 3 approches complémentaires avec pipeline temps réel et seuils adaptatifs.

## Résultats

| Modèle | AUPRC | Recall | Précision | Coût financier |
|--------|-------|--------|-----------|----------------|
| Isolation Forest | 0.049 | 0.82 | 0.01 | 94 950€ |
| Autoencoder | 0.509 | 0.89 | 0.03 | 33 120€ |
| **GNN** | **0.912** | **0.88** | **0.66** | **5 970€** |

## Dataset

- [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Télécharger `creditcard.csv` et le placer dans `data/`

## Installation
```bash
pip install numpy pandas scikit-learn imbalanced-learn pyod matplotlib seaborn jupyter torch torch-geometric joblib
```

## Structure
```
groupe-C7-fraud-detection/
├── README.md
├── data/
│   └── creditcard.csv        # À télécharger sur Kaggle
├── models/                   # Modèles sauvegardés après train.py
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_isolation_forest.ipynb
│   ├── 03_smote_preprocessing.ipynb
│   ├── 04_autoencoder.ipynb
│   ├── 05_gnn.ipynb
│   ├── 06_streaming_realtime.ipynb
│   └── 07_comparaison_finale.ipynb
├── src/
│   ├── utils.py              # Fonctions communes
│   ├── models.py             # Tous les modèles
│   ├── train.py              # Entraînement complet
│   └── predict.py            # Pipeline temps réel
└── slides/                   # Support de présentation
```

## Utilisation

### Entraîner tous les modèles
```bash
cd groupe-C7-fraud-detection
py src/train.py
```

### Lancer le pipeline temps réel
```bash
py src/predict.py
```

### Explorer les notebooks
```bash
py -m jupyter notebook
```

## Approches

### 1. Isolation Forest (baseline)
- Algorithme non-supervisé basé sur l'isolation des anomalies
- AUPRC : 0.049 → baseline de référence

### 2. Autoencoder PyTorch
- Réseau encoder-decoder entraîné sur transactions normales
- Détection par erreur de reconstruction
- AUPRC : 0.509 → 10x meilleur que la baseline

### 3. GNN — Graph Attention Network
- Graphe de transactions KNN (5 voisins)
- GAT avec 4 têtes d'attention
- AUPRC : 0.912 → niveau production 🏆

## Gestion du déséquilibre
- Dataset : 0.17% de fraudes (577:1)
- SMOTE + UnderSampling → ratio 2:1
- Class weights dans le GNN

## Pipeline temps réel
- Seuils adaptatifs selon le taux de fraude détecté
- Fenêtre glissante de 100 transactions
- Coût : 220€ sur 500 transactions

## Évaluation
- Métrique principale : **AUPRC** (adaptée aux classes déséquilibrées)
- Analyse coût financier : FP = 10€, FN = 500€
- Courbes Precision-Recall pour chaque modèle

## Références
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [PyOD](https://pyod.readthedocs.io/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [imbalanced-learn](https://imbalanced-learn.org/)