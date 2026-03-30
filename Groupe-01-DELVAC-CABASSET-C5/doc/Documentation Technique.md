# Documentation Technique — Black-Litterman Niveau Bon
**ECE Paris · Ing4 · IA Probabiliste 2026**

---

## 1. Architecture du code

Le fichier `src/bl_niveau_bon.py` est structuré en **10 sections** numérotées, chacune correspondant à une étape du pipeline :

```
Section 1  →  Téléchargement Yahoo Finance
Section 2  →  Calcul des statistiques (μ, Σ)
Section 3  →  Rendements d'équilibre (prior CAPM)
Section 4  →  Construction de la matrice Ω
Section 5  →  Formule de Black-Litterman
Section 6  →  Optimisation Monte-Carlo
Section 7  →  Frontière efficace
Section 8  →  Métriques d'un portefeuille
Section 9  →  Pipeline principal run()
Section 10 →  6 fonctions de visualisation
```

### Flux de données

```
Yahoo Finance
    │
    ▼
prices (DataFrame n_jours × n_actifs)
    │
    ├──▶ mu_hist (rendements historiques annualisés)
    └──▶ Sigma   (matrice de covariance annualisée)
                │
                ▼
           mu_eq = δ × Σ × w_mkt       ← prior CAPM (équilibre)
                │
                ▼
    ┌───────────────────────┐
    │  Vues de l'investisseur│
    │  P, Q, Ω (confiances) │
    └───────────┬───────────┘
                │
                ▼
          Black-Litterman
          μ_BL, Σ_BL             ← posterior bayésien
                │
                ├──▶ optimize_portfolio(μ_hist, Σ)  → w_Markowitz
                └──▶ optimize_portfolio(μ_BL, Σ_BL) → w_BL
                            │
                            ▼
                     6 graphiques PNG
```

---

## 2. Détail des formules mathématiques

### 2.1 Rendements d'équilibre (Prior)

On inverse le CAPM pour obtenir les rendements *implicites* du marché :

```
Π = δ × Σ × w_mkt
```

- `δ` = coefficient d'aversion au risque = 2.5 (valeur standard de Black & Litterman)
- `Σ` = matrice de covariance des actifs (annualisée)
- `w_mkt` = poids de marché (ici uniformes, simplification pédagogique)

**Intuition** : si tous les investisseurs sont rationnels et détiennent le portefeuille de marché, quels rendements justifieraient ce comportement ? C'est exactement Π.

### 2.2 Définition des vues

Chaque vue i est définie par :
- Un vecteur `pᵢ` (ligne de P) : encode "quel portefeuille" est concerné
  - Vue absolue sur actif k : `pᵢ[k] = 1`, reste à 0
  - Vue relative A sur-performe B : `pᵢ[A] = +1, pᵢ[B] = -1`, reste à 0
- Un scalaire `qᵢ` : le rendement attendu de ce portefeuille
- Un scalaire `cᵢ ∈ [0,1]` : la confiance dans cette vue

### 2.3 Calibration de la matrice Ω

Ω est la matrice diagonale d'incertitude. L'élément diagonal i est :

```
Ω_ii = ((1 - cᵢ) / cᵢ) × (pᵢ Σ pᵢᵀ) + ε
```

- `pᵢ Σ pᵢᵀ` = variance naturelle du portefeuille de la vue i
- `(1-c)/c` = facteur d'amplification qui explose quand c → 0

| Confiance | Facteur (1-c)/c | Effet |
|-----------|-----------------|-------|
| 95% | 0.05 | Vue quasi-certaine, très influente |
| 80% | 0.25 | Vue forte |
| 65% | 0.54 | Vue modérée |
| 50% | 1.00 | Vue incertaine |
| 40% | 1.50 | Vue faible |
| 10% | 9.00 | Vue presque ignorée |

### 2.4 Formule de Black-Litterman (posterior)

```
M    = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹
μ_BL = M × [(τΣ)⁻¹ × Π + PᵀΩ⁻¹ × Q]
Σ_BL = Σ + M
```

**Interprétation bayésienne** :
- `(τΣ)⁻¹ × Π` = précision du prior × valeur du prior
- `PᵀΩ⁻¹ × Q` = précision des vues × valeurs des vues
- `μ_BL` = moyenne pondérée par les précisions → donne plus de poids aux informations les plus fiables

La matrice `M` joue le rôle de variance du posterior. `Σ_BL = Σ + M` est plus grande que `Σ` car on a ajouté de l'incertitude (on ne connaît pas parfaitement μ).

### 2.5 Optimisation Monte-Carlo

On maximise le ratio de Sharpe :

```
max_w  S(w) = (wᵀμ - RF) / √(wᵀΣw)

sous :  wᵢ ≥ 0             (long only)
        Σwᵢ = 1             (budget)
        Σᵢ∈s wᵢ ≤ 0.40      (contrainte sectorielle, pour tout secteur s)
```

La méthode Monte-Carlo tire 60 000 portefeuilles avec `np.random.dirichlet` (garantit wᵢ≥0 et Σwᵢ=1), vérifie la contrainte sectorielle, et garde celui avec le Sharpe le plus élevé.

---

## 3. Choix d'implémentation et justifications

### Pourquoi Monte-Carlo plutôt que scipy.optimize ?

Pour ce projet pédagogique, Monte-Carlo a été préféré car :
1. **Lisibilité** : le code est immédiatement compréhensible, pas de boîte noire
2. **Flexibilité** : ajout de nouvelles contraintes trivial (une ligne de condition)
3. **Suffisant** : 60 000 échantillons donnent une bonne approximation pour 8 actifs
4. **Robustesse** : pas de risques de convergence numérique ou de minima locaux

En production, on utiliserait `scipy.optimize.minimize` avec `SLSQP` ou `cvxpy` pour la programmation quadratique exacte.

### Pourquoi τ = 0.05 ?

τ contrôle le poids relatif du prior par rapport aux vues. Une valeur proche de 0 signifie "je fais très confiance au prior". La règle heuristique est `τ ≈ 1/T` où T est le nombre d'années d'estimation. Avec 4 ans de données (2021-2024), τ = 1/20 = 0.05 environ.

### Pourquoi poids uniformes pour w_mkt ?

En production, on utilise les vraies capitalisations boursières (ex : poids du S&P 500). Ici, on n'a que 8 actifs, pas un indice complet. Les poids uniformes sont une simplification acceptable pour l'illustration pédagogique.

---

## 4. Description des fonctions

| Fonction | Entrées | Sorties | Description |
|----------|---------|---------|-------------|
| `get_data()` | — | DataFrame prices | Télécharge via yfinance |
| `compute_stats(prices)` | DataFrame | mu, Sigma, returns | Statistiques annualisées |
| `equilibrium_returns(Sigma)` | Sigma | np.array Π | CAPM inversé |
| `build_omega(P,Q,conf,Sigma)` | P,Q,conf,Σ | np.array Ω | Matrice d'incertitude |
| `black_litterman(mu_eq,Sigma,P,Q,Omega)` | tout | mu_bl, Sigma_bl | Posterior BL |
| `optimize_portfolio(mu,Sigma,tickers)` | mu,Σ | np.array w | Max Sharpe Monte-Carlo |
| `efficient_frontier(mu,Sigma)` | mu,Σ | vols,rets,sharpes | Nuage de portefeuilles |
| `portfolio_metrics(w,mu,Sigma)` | w,mu,Σ | ret,vol,sharpe | Métriques d'un portefeuille |
| `plot_01_rendements(...)` | data | PNG | Graphique 1 |
| `plot_02_confiance(...)` | data | PNG | Graphique 2 |
| `plot_03_allocations(...)` | data | PNG | Graphique 3 |
| `plot_04_frontiere_markowitz(...)` | data | PNG | Graphique 4 |
| `plot_05_frontiere_bl(...)` | data | PNG | Graphique 5 |
| `plot_06_recap(...)` | data | PNG | Graphique 6 |

---

## 5. Données utilisées

| Paramètre | Valeur |
|-----------|--------|
| Source | Yahoo Finance (via `yfinance`) |
| Période | 2021-01-01 → 2024-12-31 |
| Fréquence | Journalière (jours ouvrés) |
| Prix | Clôture ajustée (dividendes + splits) |
| Actifs | AAPL, MSFT, GOOGL, AMZN, JPM, GS, XOM, JNJ |
| Nb d'observations | ~1 000 jours de trading |

---

## 6. Limites et perspectives

### Limites actuelles

- **Poids de marché uniformes** : ne reflètent pas les vraies capitalisations
- **Optimisation Monte-Carlo** : approximative, pas la solution exacte
- **Vues fixes** : définies manuellement, pas mises à jour dynamiquement
- **Pas de transaction costs** : en réalité, rééquilibrer coûte des commissions

### Extensions possibles (Niveau Excellent)

- **Vues générées par ML** : signal momentum ou sentiment des news
- **Backtesting walk-forward** : simuler sur données passées avec rééquilibrage
- **Vraies capitalisations boursières** : télécharger via l'API du marché
- **Optimisation exacte** : remplacer Monte-Carlo par `cvxpy`
- **Analyse de sensibilité** : faire varier τ, δ, les confiances et observer l'impact

---

## 7. Environnement technique

```
Python   3.11
numpy    1.24+    (algèbre linéaire : inversion, produits matriciels)
pandas   2.0+     (manipulation des séries temporelles)
yfinance 0.2.38+  (API Yahoo Finance)
matplotlib 3.7+   (génération des graphiques)
```

Installation :
```bash
pip install -r requirements.txt
```
