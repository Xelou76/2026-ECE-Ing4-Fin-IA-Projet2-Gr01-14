Ce projet implémente une stratégie Wheel automatisée sur options via la plateforme QuantConnect.

L’objectif est de générer un revenu régulier en combinant la vente de puts cash-secured et de calls couverts, tout en adaptant le niveau de risque en fonction de la volatilité du marché.

## Architecture du système

L’algorithme est basé sur le framework QuantConnect (`QCAlgorithm`) et repose sur deux fonctions principales :

- `initialize()` : configuration de la stratégie  
- `on_data()` : logique exécutée à chaque arrivée de données  

##  Environnement technique

La stratégie est implémentée avec le framework QuantConnect (`QCAlgorithm`).

Elle utilise :
- les données actions et options
- les chaînes d’options (Option Chains)
- les Greeks (Delta, volatilité implicite)
- le moteur de backtesting intégré
  
##  Actifs et données
La stratégie est appliquée sur plusieurs ETF :

- IWM  
- XLF  
- XLK  

### Données utilisées

- Actions : résolution minute  
- Options : résolution horaire  

Les options sont récupérées via les **Option Chains** de QuantConnect.


##  Logique de la stratégie

La stratégie suit une machine à états :

### États possibles

- `CASH` : aucune position  
- `SHORT_PUT` : vente de put en cours  
- `LONG_STOCK` : actions détenues  
- `SHORT_CALL` : vente de call en cours  

### Transitions

- `CASH → SHORT_PUT` : vente de put  
- `SHORT_PUT → LONG_STOCK` : assignation  
- `LONG_STOCK → SHORT_CALL` : vente de call  
- `SHORT_CALL → CASH` : expiration ou vente  


## Sélection des options

### Filtrage

Les options sont filtrées selon :

- expiration entre 20 et 45 jours  
- strikes autour du prix du sous-jacent  


### Sélection par Delta

Les options sont sélectionnées selon leur Delta :

- Put : Δ ≈ -0.20  
- Call : Δ ≈ 0.25  

Si les Greeks ne sont pas disponibles, une approximation basée sur le strike est utilisée.


##  Adaptation à la volatilité

La stratégie ajuste dynamiquement les deltas en fonction de :

### 1. Volatilité implicite (IV)

- Calculée pour chaque option  
- Stockée dans un historique (`deque`)  
- Permet de calculer un percentile de volatilité  


### 2. Skew

Le skew est défini comme :

- IV put (Δ ≈ -0.25)  
- IV call (Δ ≈ 0.25)  

Formule : skew = IV_put - IV_call

### 3. Ajustement des deltas

Les deltas sont ajustés selon :

- le niveau de volatilité  
- le skew  

Cela permet d’adapter le niveau de risque :

- volatilité élevée → positions plus conservatrices  
- volatilité faible → positions plus agressives  


##  Gestion du capital

- Exposition maximale : 80% du portefeuille  
- Allocation répartie entre les actifs  
- Taille des positions calculée selon : quantité = budget / (strike × 100)


## Gestion des positions

- Les positions sont maintenues jusqu’à expiration  
- Aucune clôture anticipée  
- Gestion automatique des transitions d’état  


##  Suivi des performances

Un reporting mensuel est implémenté :

- nombre d’entrées et sorties  
- primes collectées  
- performance nette  
- valeur du portefeuille  


## Modèle de pricing

Le modèle utilisé est :

- Binomial Cox-Ross-Rubinstein (CRR)


##  Limites

- Pas de stop-loss  
- Pas de prise de profit anticipée  
- Utilisation d’ordres au marché  
- Gestion du risque simplifiée  


##  Améliorations possibles

- Ajout de règles de sortie anticipée  
- Utilisation d’ordres limites  
- Gestion avancée du risque  
- Filtrage des conditions de marché  


