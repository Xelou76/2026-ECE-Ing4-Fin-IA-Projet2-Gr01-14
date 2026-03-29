# Documentation Technique — Détection de fraude (C7)

## 1. Architecture globale

Le projet est structuré en trois modules principaux :

- Préprocessing : préparation des données
- Training : entraînement des modèles
- Inference : utilisation en production

Pipeline global :

Dataset → Preprocessing → Train → Models → Evaluation → Inference

---

## 2. Structure du projet
src/
├── preprocessing.py
├── train_baselines.py
├── train_autoencoder.py
└── inference.py


---

## 3. Préprocessing (preprocessing.py)

### Rôle

- Chargement du dataset
- Séparation des features et labels
- Normalisation des données
- Split train/test

---

### Fonctions principales

#### load_dataset(path)

Charge le fichier CSV avec pandas.

#### prepare_features(df)

- Sépare X (features)
- Sépare y (variable cible : Class)

#### split_and_scale(X, y)

- Split 80/20 (train/test)
- Standardisation des variables Time et Amount

Retour :
- X_train_scaled
- X_test_scaled
- y_train
- y_test

---

## 4. Entraînement des modèles (train_baselines.py)

### Modèles implémentés

1. Logistic Regression
   - Modèle linéaire
   - class_weight="balanced"

2. Random Forest
   - Ensemble de decision trees
   - Bonne robustesse

3. Isolation Forest
   - Non supervisé
   - Entraîné uniquement sur données normales

---

### Évaluation des modèles

Fonction principale :

evaluate_model(name, y_true, y_pred, y_score)

Métriques calculées :
- Classification report
- ROC-AUC
- PR-AUC
- Matrice de confusion
- Courbe Precision-Recall
- Coût financier

---

### Analyse du coût

Fonction :

compute_cost(y_true, y_pred, cost_fp=1, cost_fn=200)

Retour :
- FP (False Positives)
- FN (False Negatives)
- Coût total

---

### Sauvegarde des modèles

Les modèles sont sauvegardés dans le dossier artifacts/ :

- logistic_regression.joblib
- random_forest.joblib
- isolation_forest.joblib

---

## 5. Autoencoder (train_autoencoder.py)

### Architecture

Réseau de neurones :

Input → 16 → 8 → 4 → 8 → 16 → Output

- Encoder : compression
- Decoder : reconstruction

---

### Principe

- Entraîné uniquement sur les transactions normales
- Détection via erreur de reconstruction

---

### Calcul du seuil

Seuil basé sur le percentile 95 :

threshold = np.percentile(train_error, 95)

---

### Métriques

- ROC-AUC
- PR-AUC
- Matrice de confusion
- Analyse coût

---

### Sauvegarde

- autoencoder.pt
- autoencoder_loss.png
- autoencoder_cm.png
- autoencoder_pr_curve.png

---

## 6. Inférence (inference.py)

### Objectif

Simuler un système de détection en production.

---

### Fonctionnement

1. Chargement des modèles depuis artifacts/
2. Prédiction sur une transaction
3. Calcul du score de fraude
4. Mesure de la latence

---

### Latence

Mesurée en millisecondes :

latency = (time.time() - start) * 1000

---

### Simulation streaming

- Traitement de plusieurs transactions
- Affichage :
  - true label
  - prédiction
  - score
  - latence

Exemple :

[00] true=0 pred=0 score=0.000039 latency=30 ms

---

## 7. Gestion du déséquilibre

Méthodes utilisées :

- class_weight="balanced"
- entraînement uniquement sur données normales (Autoencoder, Isolation Forest)

---

## 8. Métriques utilisées

### ROC-AUC
Mesure la capacité à séparer les classes.

### PR-AUC
Plus adaptée aux datasets déséquilibrés.

### Matrice de confusion
Analyse détaillée :
- FP
- FN

---

## 9. Analyse coût

Hypothèses :

- Faux positif = 1€
- Faux négatif = 200€

Objectif :
minimiser le coût total plutôt que maximiser uniquement la précision.

---

## 10. Stockage des artefacts

Dossier :

artifacts/

Contient :
- modèles entraînés
- matrices de confusion
- courbes PR
- courbes de loss

---

## 11. Performances

Latence observée :

- Logistic Regression : ~3 ms
- Random Forest : ~30–40 ms
- Isolation Forest : ~10 ms
- Autoencoder : ~20 ms

---

## 12. Limites

- Simulation temps réel simplifiée
- Seuil fixe (autoencoder)
- Pas de GNN
- Dataset anonymisé

---

## 13. Améliorations possibles

- Optimisation du threshold
- SMOTE / techniques avancées de rééquilibrage
- GNN (graphes de transactions)
- Pipeline streaming réel
- Dashboard de monitoring

---

## 14. Technologies

- Python
- pandas
- numpy
- scikit-learn
- PyTorch
- matplotlib
- joblib