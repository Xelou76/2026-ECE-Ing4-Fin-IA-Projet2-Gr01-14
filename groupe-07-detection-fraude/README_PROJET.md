# C.7 — Détection de fraude en temps réel avec Machine Learning

Projet pédagogique ECE Ing4 — Détection de fraude bancaire sur le dataset **Kaggle Credit Card Fraud Detection** à l’aide de plusieurs approches de machine learning supervisées et non supervisées.

## Objectif

L’objectif du projet est de comparer plusieurs méthodes de détection de fraude dans un contexte de **fort déséquilibre de classes** et de **contrainte de latence**.

Nous avons implémenté et comparé :

- **Logistic Regression**
- **Random Forest**
- **Isolation Forest**
- **Autoencoder**

Le projet inclut :
- l’entraînement des modèles
- l’évaluation avec des métriques adaptées
- une **analyse métier par coût financier**
- une **simulation d’inférence temps réel**

---

## Problématique

La détection de fraude est un problème difficile car :

- les fraudes sont **très rares**
- les données sont **fortement déséquilibrées**
- les erreurs n’ont pas le même coût :
  - un **faux positif** bloque une transaction légitime
  - un **faux négatif** laisse passer une fraude

Dans ce contexte, l’évaluation ne peut pas se limiter à l’accuracy. Nous utilisons notamment :

- **ROC-AUC**
- **PR-AUC**
- **matrice de confusion**
- **coût financier des erreurs**

---

## Dataset

Dataset utilisé : **Kaggle Credit Card Fraud Detection**

- Nombre total de transactions : **284 807**
- Nombre de fraudes : **492**
- Taux de fraude : **0.17%**

Colonnes :
- `V1` à `V28` : variables anonymisées
- `Time`
- `Amount`
- `Class` : variable cible
  - `0` = transaction normale
  - `1` = fraude

Le fichier attendu est :

```text
data/creditcard.csv
________________________________________
Structure du projet
groupe-C7-Detection-fraude/
├── artifacts/
│   ├── autoencoder.pt
│   ├── autoencoder_cm.png
│   ├── autoencoder_loss.png
│   ├── autoencoder_pr_curve.png
│   ├── isolation_forest.joblib
│   ├── isolation_forest_cm.png
│   ├── isolation_forest_pr_curve.png
│   ├── logistic_regression.joblib
│   ├── logistic_regression_cm.png
│   ├── logistic_regression_pr_curve.png
│   ├── random_forest.joblib
│   ├── random_forest_cm.png
│   └── random_forest_pr_curve.png
├── data/
│   └── creditcard.csv
├── src/
│   ├── preprocessing.py
│   ├── train_baselines.py
│   ├── train_autoencoder.py
│   └── inference.py
├── requirements.txt
└── README.md
________________________________________
Installation
1. Cloner le projet
git clone <url-du-repo>
cd groupe-C7-Detection-fraude
2. Créer un environnement virtuel
Windows PowerShell
py -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Installer les dépendances
Windows
py -m pip install -r requirements.txt
Linux / macOS
python -m pip install -r requirements.txt
________________________________________
Utilisation
1. Télécharger le dataset
Télécharger le dataset Kaggle puis placer le fichier creditcard.csv dans :
data/creditcard.csv
2. Entraîner les modèles classiques
py src/train_baselines.py
Ce script :
•	charge et prépare les données 
•	entraîne : 
o	Logistic Regression 
o	Random Forest 
o	Isolation Forest 
•	calcule les métriques 
•	calcule le coût financier 
•	sauvegarde les modèles et figures dans artifacts/ 
3. Entraîner l’autoencoder
py src/train_autoencoder.py
Ce script :
•	entraîne un autoencoder sur les transactions normales 
•	calcule l’erreur de reconstruction 
•	applique un seuil de détection 
•	génère les métriques et graphiques 
•	sauvegarde le modèle dans artifacts/ 
4. Lancer l’inférence
py src/inference.py
Ce script :
•	recharge les modèles sauvegardés 
•	effectue une prédiction sur une transaction 
•	affiche : 
o	la vraie classe 
o	la classe prédite 
o	le score de fraude 
o	le seuil utilisé 
o	la latence 
•	simule un flux de transactions en quasi temps réel 
________________________________________
Préprocessing
Le fichier src/preprocessing.py gère :
•	le chargement du dataset 
•	la séparation X / y 
•	le découpage train / test 
•	la standardisation des colonnes Time et Amount 
Les modèles supervisés sont entraînés sur l’ensemble du train set.
Les modèles non supervisés :
•	Isolation Forest 
•	Autoencoder 
sont entraînés uniquement sur les transactions normales.
________________________________________
Tests / Exécution attendue
Test 1 — Entraînement baseline
Commande :
py src/train_baselines.py
Résultat attendu :
•	affichage des métriques pour 3 modèles 
•	création des fichiers : 
o	logistic_regression.joblib 
o	random_forest.joblib 
o	isolation_forest.joblib 
o	matrices de confusion 
o	courbes PR 
Test 2 — Entraînement autoencoder
Commande :
py src/train_autoencoder.py
Résultat attendu :
•	affichage de la loss par epoch 
•	affichage des métriques du modèle 
•	création de : 
o	autoencoder.pt 
o	autoencoder_loss.png 
o	autoencoder_cm.png 
o	autoencoder_pr_curve.png 
Test 3 — Inférence
Commande :
py src/inference.py
Résultat attendu :
•	affichage d’une prédiction unitaire 
•	simulation de flux sur plusieurs transactions 
•	affichage de la latence moyenne 
________________________________________
Résultats obtenus
Tableau comparatif
Modèle	PR-AUC	FP	FN	Coût total
Logistic Regression	0.7159	1386	8	2986
Random Forest	0.8291	15	18	3615
Autoencoder	0.4786	2886	13	5486
Isolation Forest	0.1283	100	72	14500
Interprétation
•	Random Forest obtient la meilleure PR-AUC 
•	Logistic Regression obtient le coût total le plus faible 
•	Autoencoder détecte bien les fraudes mais génère trop de faux positifs 
•	Isolation Forest est la méthode la moins performante sur ce dataset 
________________________________________
Analyse métier
Nous avons introduit un modèle de coût simple :
•	Faux positif (FP) : 1 € 
•	Faux négatif (FN) : 200 € 
Cela permet de rapprocher l’évaluation des contraintes métier réelles :
•	un faux positif entraîne un coût opérationnel et une mauvaise expérience utilisateur 
•	un faux négatif entraîne une perte financière directe 
Cette analyse montre que le meilleur modèle selon les métriques classiques n’est pas forcément le meilleur d’un point de vue business.
________________________________________
Simulation temps réel
Le fichier inference.py permet de simuler un scénario de détection transaction par transaction.
Exemple d’éléments mesurés :
•	score de fraude 
•	classe prédite 
•	temps d’inférence 
•	latence moyenne 
Sur notre test, la latence observée est compatible avec une détection quasi temps réel à petite échelle.
________________________________________
Choix du meilleur modèle
Deux conclusions sont possibles selon l’objectif :
Si on privilégie la performance ML
Le Random Forest est le meilleur modèle grâce à sa meilleure PR-AUC et son très faible nombre de faux positifs.
Si on privilégie le coût métier
La Logistic Regression est le meilleur choix car elle minimise le coût total en réduisant fortement les fraudes non détectées.
________________________________________
Limites du projet
•	dataset anonymisé, peu interprétable métier 
•	coût financier basé sur une hypothèse simplifiée 
•	simulation temps réel à petite échelle 
•	pas d’implémentation GNN 
•	pas d’optimisation dynamique du seuil 
________________________________________
Perspectives
•	optimisation du threshold pour minimiser le coût 
•	utilisation de SMOTE ou d’autres techniques de rééquilibrage 
•	implémentation d’un GNN sur un graphe de transactions 
•	pipeline streaming plus réaliste 
•	dashboard de monitoring 
________________________________________
Technologies utilisées
•	Python 
•	pandas 
•	numpy 
•	scikit-learn 
•	PyTorch 
•	matplotlib 
•	joblib 
________________________________________
Références
•	Kaggle Credit Card Fraud Detection Dataset 
•	scikit-learn documentation 
•	PyTorch documentation 
•	PyOD documentation 
•	imbalanced-learn documentation

