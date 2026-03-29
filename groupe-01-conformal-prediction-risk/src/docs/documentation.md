# Documentation Technique : Conformal Prediction 

## 1. Problématique : L'Incertitude en Finance
En finance quantitative, un modèle qui donne une seule valeur est risqué car il ne précise pas sa marge d'erreur. 

La Conformal Prediction  résout ce problème. 
C'est une méthode statistique qui permet de créer un "tunnel de confiance" autour de la prédiction. 
Elle garantit mathématiquement que le prix réel restera dans ce tunnel dans 95% des cas.

## 2. Fonctionnement de l'Algorithme (Split-Conformal)

Nous utilisons une approche en trois étapes pour garantir la fiabilité du modèle :

### Étape A : Séparation des données
On divise les données historiques en deux parties :
Set d'Entraînement : Pour apprendre au modèle Random Forest à prédire les prix.
Set de Calibration : Des données "neuves" pour mesurer l'erreur réelle du modèle.

### Étape B : Calcul du Score de Non-Conformité
Pour chaque donnée du set de calibration, on calcule l'erreur du modèle :
Score = Valeur Absolue de (Prix Réel - Prédiction)

Ce score représente à quel point le modèle se trompe sur des données qu'il ne connaît pas.

### Étape C : Création de l'Intervalle de Confiance
Pour un niveau de confiance de 95% :
1. On cherche le seuil d'erreur  qui couvre 95% de nos scores de calibration.
2. L'intervalle final est alors : **[Prédiction - Seuil d'erreur, Prédiction + Seuil d'erreur]**.



## 3. Pourquoi est-ce utile pour le Risk Management ?

1. Adaptabilité à la Volatilité: Si le marché devient nerveux (forte volatilité), les erreurs du modèle augmentent. Le "tunnel" rouge s'élargit alors automatiquement pour continuer à couvrir 95% des cas. C'est un excellent indicateur visuel de risque.
2. Indépendance des Lois de Probabilité : Contrairement aux modèles classiques qui supposent que les prix suivent une courbe "normale" (Gaussienne), la Conformal Prediction fonctionne quelle que soit la forme des données. C'est idéal pour les krachs boursiers ou les événements extrêmes.
3. Garantie Mathématique: C'est l'un des rares outils de Machine Learning qui offre une preuve statistique de sa fiabilité sur le long terme.

## 4. Structure du Code
* Bibliothèques : `yfinance` (données), `scikit-learn` (Random Forest), `MAPIE` (calcul de l'intervalle).
* Entrée : Prix de clôture du S&P 500 sur les 3 derniers jours.
* Sortie : Prédiction pour le lendemain avec son intervalle de risque à 95%.
