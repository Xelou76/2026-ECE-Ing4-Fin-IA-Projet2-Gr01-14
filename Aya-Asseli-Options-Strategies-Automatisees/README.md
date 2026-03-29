# Stratégie Wheel automatisée multi-actifs

Ce projet implémente une stratégie **Wheel automatisée** sur options via QuantConnect.

L’objectif est de générer un revenu régulier en vendant des options (puts et calls) sur des ETF liquides.


##  Actifs utilisés

- IWM (Russell 2000)
- XLF (secteur financier)
- XLK (secteur technologique)


##  Principe de la stratégie

La stratégie Wheel suit un cycle :

1. Vente de put cash-secured  
2. Si assignation → achat de l’actif  
3. Vente de call couvert  
4. Répétition du cycle  


##  Fonctionnement

L’algorithme utilise une machine à états :

- `CASH` : vente de put  
- `SHORT_PUT` : attente de l’expiration  
- `LONG_STOCK` : vente de call  
- `SHORT_CALL` : attente de l’expiration  


##  Sélection des options

- Maturité : entre 20 et 45 jours  
- Sélection basée sur le Delta :
  - Put ≈ -0.20  
  - Call ≈ 0.25  

Les deltas sont ajustés dynamiquement en fonction :
- de la volatilité implicite (IV)
- du skew (puts vs calls)


##  Gestion du capital

- Exposition limitée à 80% du portefeuille  
- Allocation répartie entre les actifs  
- Taille des positions basée sur le capital disponible  


##  Suivi des performances

Un reporting mensuel permet de suivre :
- le nombre de trades  
- les primes collectées  
- la performance du portefeuille  


## Détails techniques

- Plateforme : QuantConnect  
- Langage : Python  
- Données :
  - Actions : minute  
  - Options : heure  


##  Installation

1. Créer un compte sur QuantConnect  
2. Créer un projet Python  
3. Copier le code dans le fichier `main.py`  
4. Lancer le backtest


##  Utilisation

1. Ouvrir le projet sur QuantConnect  
2. Lancer le backtest  
3. Observer :
   - les trades (puts et calls)
   - les transitions d’état
   - la performance du portefeuille  

Les logs permettent de suivre :
- les entrées en position  
- les changements d’état  
- les résultats mensuels  

##  Tests

- Backtest sur la période 2024 – 2026  
- Vérification du bon fonctionnement de la stratégie :
  - vente de puts et calls  
  - transitions entre les états  
  - calcul des tailles de position  

Résultats observables :
- nombre de trades  
- primes collectées  
- évolution du portefeuille  
