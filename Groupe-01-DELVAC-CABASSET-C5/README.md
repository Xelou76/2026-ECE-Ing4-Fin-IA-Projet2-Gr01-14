# Optimisation de Portefeuille Bayésien – Black-Litterman
**ECE Paris · Ing4 · IA Probabiliste, Théorie des Jeux & ML · 2026**

> **Sujet C.5** · Difficulté 3/5 · Domaine : Probas, ML · **Niveau Bon**

---

## Contexte du projet

La théorie de Markowitz (1952) est le fondement de la gestion de portefeuille moderne : elle calcule les poids optimaux qui maximisent le rendement pour un niveau de risque donné. Mais elle souffre d'un défaut majeur — elle dépend entièrement des **rendements historiques**, qui sont instables et bruités.

**Black-Litterman (Goldman Sachs, 1992)** résout ce problème en adoptant une approche **bayésienne** :

- **Prior** : les rendements d'équilibre du marché (CAPM inversé) — ce que le marché *dit implicitement* par ses prix
- **Likelihood** : les **vues** de l'investisseur — ce qu'il *pense* du futur, avec un niveau de confiance
- **Posterior** : les rendements Black-Litterman — la fusion optimale des deux

```
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹ · Π + PᵀΩ⁻¹ · Q]
```

| Symbole | Signification |
|---------|---------------|
| `Π`     | Rendements d'équilibre (prior CAPM) |
| `P`     | Matrice des vues (k×n) — quels actifs sont concernés |
| `Q`     | Valeurs des vues — rendements attendus |
| `Ω`     | Incertitude sur les vues (calibrée par la confiance) |
| `τ`     | Confiance dans le prior (~1/nb années) |

---

## Objectifs atteints (Niveau Bon)

| Objectif | Implémentation | Statut |
|----------|----------------|--------|
| Vues avec niveaux de confiance variables | Matrice Ω calibrée par `(1-c)/c × pᵢΣpᵢᵀ` | ✅ |
| Optimisation sous contraintes | Monte-Carlo : long-only + budget + secteur ≤ 40% | ✅ |
| Frontière efficace Markowitz vs BL | 10 000 portefeuilles aléatoires + point optimal | ✅ |

---

## Installation

### Prérequis

- Python **3.11** (recommandé)
- PyCharm ou tout autre IDE
- Connexion internet (pour Yahoo Finance)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/CoursIA.git
cd CoursIA/groupe-XX-portfolio-bayesien

# 2. Créer un environnement virtuel
python -m venv venv

# Activer — Windows
venv\Scripts\activate

# Activer — macOS/Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

```bash
python src/bl_niveau_bon.py
```

Le programme affiche les résultats dans la console et génère **6 graphiques PNG** dans `results/` :

| Fichier | Contenu |
|---------|---------|
| `01_rendements_prior_posteriori.png` | Comparaison Historique / Équilibre CAPM / BL Posteriori |
| `02_vues_confiance.png` | Vues de l'investisseur colorées par niveau de confiance |
| `03_allocations.png` | Poids Markowitz vs BL côte à côte |
| `04_frontiere_markowitz.png` | Frontière efficace Markowitz (rendements historiques) |
| `05_frontiere_bl.png` | Frontière efficace Black-Litterman (rendements posteriori) |
| `06_recap.png` | Tableau comparatif des métriques de performance |

---

## Structure du projet

```
groupe-XX-portfolio-bayesien/
├── README.md                  ← ce fichier
├── requirements.txt           ← dépendances Python
├── src/
│   └── bl_niveau_bon.py       ← code source principal (commenté)
├── results/                   ← graphiques générés (créé automatiquement)
│   ├── 01_rendements_prior_posteriori.png
│   ├── 02_vues_confiance.png
│   ├── 03_allocations.png
│   ├── 04_frontiere_markowitz.png
│   ├── 05_frontiere_bl.png
│   └── 06_recap.png
└── docs/
    └── documentation_technique.md
```

---

## Paramètres configurables

Dans `src/bl_niveau_bon.py`, tout en haut du fichier :

```python
TICKERS = ["AAPL", "MSFT", ...]   # actifs à inclure
START   = "2021-01-01"             # début de la période historique
END     = "2024-12-31"             # fin de la période historique
RF      = 0.04                     # taux sans risque (bonds du Trésor US)
TAU     = 0.05                     # confiance dans le prior (≈ 1/nb années)
DELTA   = 2.5                      # aversion au risque du marché
```

Les **vues** se définissent dans la fonction `run()` :

```python
abs_view("AAPL", 0.10, 0.80, "AAPL +10% (conf 80%)")   # vue absolue
rel_view("JPM", "GOOGL", 0.04, 0.40, "JPM > GOOGL")    # vue relative
```

---

## Dépendances

```
numpy>=1.24        # calcul matriciel
pandas>=2.0        # manipulation des données
matplotlib>=3.7    # graphiques
yfinance>=0.2.38   # données boursières réelles
```

---

## Membres du groupe

| Nom Prénom | GitHub |
|------------|--------|
| Prénom NOM | @username |
| Prénom NOM | @username |
| Prénom NOM | @username |

---

## Références

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal, 48(5), 28–43.
- [PyPortfolioOpt – Black-Litterman](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html)
- [Wikipedia – Black-Litterman model](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model)
- [Thomas Starke – BL sur QuantConnect](https://www.quantconnect.com/learning/articles/introduction-to-black-litterman)
