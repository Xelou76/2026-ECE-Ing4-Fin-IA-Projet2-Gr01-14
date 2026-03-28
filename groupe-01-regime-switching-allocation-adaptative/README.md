# Stratégie d'Investissement Adaptative : Regime Switching (QuantConnect)

Bienvenue dans le dépôt du projet **H.4 - Regime Switching et Allocation Adaptative**. 

Ce projet propose une stratégie de trading algorithmique de niveau "Excellent" développée sur **QuantConnect**. Elle utilise des modèles de Machine Learning développés de zéro (en pur `numpy`) pour détecter les régimes de marché, Bull, Neutral, et Bear, puis adapter l'allocation d'un portefeuille multi-actifs de manière dynamique.

## Fonctionnalités Principales

*   **Détection par Machine Learning (Pur NumPy)** : Modèle de Markov Caché (HMM) et algorithme K-Means codés sans librairies externes  pour une bonne compatibilité avec QuantConnect.
*   **Allocation Multi-Actifs** : Actions (SPY), Obligations (TLT, IEF), Or (GLD) et Matières Premières (DJP).
*   **Filet de Sécurité Quotidien** : Coupe-circuit basé sur la volatilité à court terme pour forcer un passage en régime défensif en cas de krach soudain.
*   **Vote Majoritaire** : Combinaison des prédictions HMM et K-Means avec un biais de prudence en cas de désaccord.

## 1. INSTALLATION

Ce projet est conçu pour s'exécuter sur le moteur de backtest **QuantConnect (Lean Engine)**. Voila une option pour l'installer et l'exécuter :

### Via la Local Plateforme QuantConnect

1.  Créez un compte sur [QuantConnect](https://www.quantconnect.com/).
2.  Créez un nouveau projet algorithmique en sélectionnant le langage **Python**.
3.  Copiez l'intégralité du code fourni dans ce dépôt et collez-le dans le fichier `main.py` de votre projet QuantConnect.
4.  Assurez-vous que les imports initiaux sont bien présents :

    ```python
    from AlgorithmImports import *
    import numpy as np
    ```

## 2. Utilisation

Une fois le code installé, la stratégie fonctionne de manière entièrement autonome selon le cycle suivant :

1.  **Phase de Warm-up** : Au lancement, l'algorithme télécharge 300 jours d'historique pour initialiser les indicateurs.

2.  **Entraînement Mensuel** : Le premier jour de chaque mois, l'algorithme :
    *   Extrait les 400 derniers jours de données du SPY.
    *   Entraîne le HMM et le K-Means.
    *   Détecte le régime actuel (Bull, Neutral, ou Bear).
    *   Liquide et réalloue le portefeuille selon la matrice d'allocation définie dans `ALLOC`.

3.  **Surveillance Quotidienne** : Chaque jour, la fonction `_daily_crisis_check` vérifie la volatilité sur 5 jours. Si elle dépasse 25%, le portefeuille passe immédiatement en régime "Bear" d'urgence.

## 3. Tests et Backtesting

Pour évaluer les performances de la stratégie, vous devez lancer un **Backtest**. 

### Paramètres par défaut configurés dans le code :

*   **Période de test** : Du `1er Janvier 2010` au `1er Janvier 2024`. (Couvre plusieurs crises majeures comme le krach COVID de 2020 et le marché baissier de 2022).
*   **Capital initial** : `100 000 $`.
*   **Brokerage Model** : Interactive Brokers (compte sur marge).
*   **Benchmark** : SPY (S&P 500). 

### Lancer le test :

*   **Sur le Web IDE** : Cliquez simplement sur le bouton **"Backtest"** en haut à droite de l'interface QuantConnect.
*   **En local (Lean CLI)** : Exécutez la commande suivante dans votre terminal :

    ```bash
    lean backtest "RegimeSwitching"
    ```

### Résultats attendus lors du test :

Dans la console de logs (onglet *Log* de QuantConnect), vous verrez l'algorithme imprimer ses décisions :

*   **Accord des modèles :** `2020-05-01 | HMM=bull KM=bull → BULL`
*   **Désaccord (Prudence appliquée) :** `2022-02-01 | HMM=neutral KM=bear → BEAR`
*   **Déclenchement du coupe-circuit :** `2020-03-09 | CRISE vol5=32.4% → BEAR forcé`

Analysez l'onglet *Equity* et *Drawdown* de QuantConnect à la fin du test pour constater la réduction des pertes lors des marchés baissiers par rapport au Benchmark (SPY). 