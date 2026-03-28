# Documentation Technique : Stratégie Regime Switching (QuantConnect)

Ce document détaille l'architecture logicielle, les algorithmes de Machine Learning implémentés et la logique de trading de la stratégie `RegimeSwitchingExcellent`.

## 1. Architecture Logicielle

Le code est structuré en deux grandes parties mathématiques et logiques :

1. **Moteur d'Intelligence Artificielle (Pur NumPy)** : Deux classes indépendantes (`KMeans` et `GaussianHMM`) développées sans dépendances externes pour s'assurer d'une exécution optimisée dans le cloud QuantConnect.

2. **Logique Algorithmique (QuantConnect)** : Une classe héritant de `QCAlgorithm` qui orchestre la récupération des données, la préparation des "features", l'entraînement des modèles, la prédiction et l'exécution des ordres.

---

## 2. Moteur de Machine Learning

### 2.1. Classe `KMeans`
Implémentation de l'algorithme K-Means en utilisant l'algorithme de Lloyd.

* **Initialisation** : `__init__(self, n_clusters=3, n_iter=100, n_init=5)`
  * `n_clusters` : Nombre de régimes à détecter (3 : Bull, Neutral, Bear).
  * `n_iter` : Nombre d'itérations maximales pour la convergence.
  * `n_init` : Nombre de fois où l'algorithme est exécuté avec des centroïdes initiaux différents pour éviter les minimums locaux.
 
* **Méthodes Principales** :
  * `fit(self, X)` : Entraîne le modèle sur la matrice de caractéristiques `X`. Calcule la distance euclidienne entre les points et les centroïdes, réassigne les étiquettes et met à jour les centroïdes jusqu'à convergence ou au bout de `n_iter`. Conserve la meilleure exécution basée sur l'inertie (somme des distances au carré).
  * `predict(self, X)` : Assigne chaque nouvelle observation au centroïde le plus proche.

### 2.2. Classe `GaussianHMM`
Implémentation d'un Modèle de Markov Caché (HMM) à émissions gaussiennes. Ce modèle suppose que le marché évolue selon des états cachés, des régimes et que les observations (rendements, volatilité) suivent une distribution normale spécifique à chaque état.

* **Initialisation** : `__init__(self, n_states=3, n_iter=60, tol=1e-4)`

* **Méthodes et Algorithmes Sous-jacents** :
  * `_init_params(self, X)` : Initialise les probabilités initiales (pi), la matrice de transition (A) en favorisant la persistance des états (0.95 sur la diagonale), ainsi que les moyennes (mu) et variances (sig) par terciles des données triées.

  * `_forward` et `_backward` : Implémentation de l'algorithme Forward-Backward. Utilise la fonction `np.logaddexp` pour calculer les probabilités dans l'espace logarithmique. Cela empêche les erreurs d'underflow fréquentes sur de longues séries temporelles.

  * `fit(self, X)` : Algorithme Espérance-Maximisation, Baum-Welch. Alterne entre l'estimation des probabilités d'état (Étape E) et la mise à jour des paramètres $$\mu$$, $$\sigma$$, et des matrices de transition (Étape M) jusqu'à ce que l'amélioration de la log-vraisemblance soit inférieure à la tolérance (`tol`).

  * `predict(self, X)` : Algorithme de Viterbi. Décode la séquence d'états cachés la plus probable compte tenu des nouvelles observations.

---

## 3. Classe Principale : `RegimeSwitchingExcellent`

Cette classe gère le cycle de vie de la stratégie dans QuantConnect.

### 3.1. Gestion du Temps et des Événements (`Initialize`)
* **Univers d'investissement** : Actions (SPY), Obligations à long terme (TLT), Obligations à moyen terme (IEF), Or (GLD), Matières premières (DJP).
  
* **Scheduling** : 
  * `_monthly_rebalance` : Exécuté le premier jour de chaque mois, 30 minutes après l'ouverture.
  * `_daily_crisis_check` : Exécuté tous les jours, 60 minutes après l'ouverture.

### 3.2. Ingénierie des Caractéristiques (`_get_features`)
La préparation des données est cruciale pour le ML. La fonction extrait un historique de 400 jours de clôture du SPY et calcule 4 dimensions (`X_raw`) :
1. **Rendement journalier** : Différence des logarithmes des prix.
2. **Volatilité à 20 jours** : Écart-type glissant des rendements.
3. **Momentum à 60 jours** : Taux de variation du prix sur 60 jours.
4. **Ratio de volatilité** : Volatilité 20 jours divisée par la volatilité 60 jours.

**Normalisation Robuste** :
Pour éviter que les krachs extrêmes ne faussent les algorithmes, les données subissent un "clipping" basé sur le Median Absolute Deviation.
$$MAD = median(|X_i - median(X)|) * 1.4826$$
Les valeurs sont bornées à $$+/- 4 * MAD$$, puis centrées-réduites (Z-Score).

### 3.3. Définition et Étiquetage des Régimes (`_label_states`)
Le ML non-supervisé retourne des clusters arbitraires (0, 1, 2). La stratégie doit leur donner un sens financier. L'algorithme d'étiquetage procède ainsi :
1. **Bear (Baissier)** : Le cluster avec la volatilité moyenne la plus élevée.
2. **Bull (Haussier)** : Parmi les clusters restants, celui avec le rendement moyen le plus élevé.
3. **Neutral (Neutre)** : Le cluster restant.

### 3.4. Logique de Vote Majoritaire (`_detect_regime`)
Avant chaque rebalancement mensuel, le HMM et le K-Means classifient les 30 derniers jours. L'état du marché est défini par l'étiquette du dernier jour.
* Si HMM et K-Means s'accordent, l'état est validé.
* En cas de désaccord, un dictionnaire de priorité est appliqué (`"bear": 0, "neutral": 1, "bull": 2`). L'algorithme sélectionne l'état le plus défensif pour minimiser le risque de drawdown.

### 3.5. Exécution et Filet de Sécurité

* **`_monthly_rebalance`** : Liquide le portefeuille et applique les pondérations de la matrice `ALLOC` en fonction du régime détecté. Le régime "Bear" inclut le DJP (Commodities) comme protection contre l'inflation, tandis que le "Bull" est massivement investi en actions (85%).

* **`_daily_crisis_check`** : Mécanisme de coupe-circuit. Si la volatilité annualisée des 5 derniers jours (`vol5`) dépasse 25% (`CRISIS_VOL_THRESHOLD`), le système annule l'allocation actuelle, force le régime en "bear" de manière anticipée et réalloue immédiatement le capital vers les valeurs refuges. Le régime normal reprend lors du rebalancement mensuel si la volatilité redescend sous le seuil.

---

## 4. Complexité et Performances
* **Complexité Spatiale** : Faible. La matrice d'entraînement est limitée à 400 lignes et 4 colonnes.
* **Complexité Temporelle** : L'entraînement des modèles (K-Means + HMM) prend quelques secondes par mois. La prédiction quotidienne (`predict` Viterbi) est en  $$O(T * K^2)$$ où T est la séquence, 30 jours et K le nombre d'états : 3, soit une exécution quasi instantanée.
