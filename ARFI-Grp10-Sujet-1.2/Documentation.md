# Documentation technique — Bayesian Sports Analytics Ligue 1

**Projet** : I.2 — Modèles hiérarchiques bayésiens pour prédire les résultats sportifs  
**Cours** : ECE Paris — Ing4 — IA Probabiliste, Théorie des Jeux et Machine Learning  
**Auteur** : Ethan Arfi  


## 1. Contexte et objectif

Ce projet construit un modèle bayésien hiérarchique pour prédire les résultats de Ligue 1, puis compare les probabilités du modèle aux cotes Bet365 pour identifier des value bets.

Le modèle implémenté est celui de Baio & Blangiardo (2010), référence académique en statistiques sportives bayésiennes.



## 2. Architecture du modèle

### 2.1 Formulation mathématique

Les buts marqués par chaque équipe suivent une loi de Poisson :

```
log(theta_domicile) = home_adv + att[h] + def[a]
log(theta_exterieur) =           att[a] + def[h]

buts_domicile  ~ Poisson(theta_domicile)
buts_exterieur ~ Poisson(theta_exterieur)
```

### 2.2 Paramètres et priors

| Paramètre | Distribution | Signification |
|-----------|-------------|---------------|
| `home_adv` | Normal(0, 1) | Avantage domicile — estimé à +0.32 |
| `sigma_att` | HalfNormal(1) | Dispersion des attaques entre équipes |
| `sigma_def` | HalfNormal(1) | Dispersion des défenses entre équipes |
| `att[t]` | ZeroSumNormal(sigma_att) | Force d'attaque de l'équipe t |
| `def[t]` | ZeroSumNormal(sigma_def) | Faiblesse défensive de l'équipe t |

### 2.3 Contrainte somme-nulle

Le `ZeroSumNormal` garantit que la somme des forces d'attaque (et de défense) sur toutes les équipes est nulle. Sans cette contrainte, le modèle ne peut pas distinguer "toutes les équipes sont fortes" de "toutes les équipes sont faibles" — `home_adv` absorberait l'ambiguïté.

---

## 3. Données

### 3.1 Source

Données publiques depuis [football-data.co.uk](https://www.football-data.co.uk/frenchm.php), téléchargées automatiquement à l'exécution.

| Saison | Matchs | Utilisation |
|--------|--------|-------------|
| 2022-23 | 380 matchs | Modèle dynamique uniquement |
| 2023-24 | 306 matchs | Modèle principal, calibration, value bets |
| Total | 686 matchs | EDA (exploration des données) |

La saison 2023-24 contient 306 matchs (et non 380) car les données ont été téléchargées avant la fin de la saison.

### 3.2 Variables principales

- `FTHG` / `FTAG` : buts domicile / extérieur en fin de match
- `FTR` : résultat final (H = domicile, D = nul, A = extérieur)
- `B365H/D/A` : cotes Bet365 (marge moyenne ~5.4%)
- `PSH/PSD/PSA` : cotes Pinnacle (marge ~2%, bookmaker de référence)

---

## 4. Inférence MCMC

### 4.1 Algorithme NUTS

L'inférence utilise NUTS (No-U-Turn Sampler), variante adaptative du Monte Carlo hamiltonien, implémentée dans PyMC.

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `draws` | 2000 | Échantillons par chaîne après adaptation |
| `tune` | 1000 | Étapes d'adaptation (écartées) |
| `chains` | 4 | Chaînes indépendantes pour le diagnostic |
| `target_accept` | 0.9 | Recommandé pour les modèles hiérarchiques |
| `random_seed` | 42 | Reproductibilité |

### 4.2 Résultats de convergence

| Paramètre | Moyenne | HDI 3% | HDI 97% | R̂ | ESS |
|-----------|---------|--------|---------|-----|-----|
| `home_adv` | 0.319 | 0.226 | 0.408 | 1.001 | 10568 |
| `sigma_att` | 0.274 | 0.156 | 0.399 | 1.000 | 6299 |
| `sigma_def` | 0.193 | 0.085 | 0.313 | 1.001 | 2319 |

R̂ maximum : **1.0018** — convergence excellente (seuil critique : 1.01).  
ESS largement au-dessus du minimum recommandé de 400 sur tous les paramètres.

---

## 5. Résultats

### 5.1 Paramètres globaux

- **home_adv = 0.319** : jouer à domicile augmente les buts d'environ 37% (exp(0.319) ≈ 1.37). L'intervalle de crédibilité à 94% ne contient pas 0.
- **sigma_att = 0.274** : les équipes se différencient modérément en attaque.
- **sigma_def = 0.193** : les défenses sont légèrement moins dispersées que les attaques.

### 5.2 Forces des équipes

Le modèle retrouve automatiquement la hiérarchie réelle de la Ligue 1 2023-24 à partir des seuls scores — sans classement ni information externe. PSG domine en attaque, Nice et Lille ressortent comme meilleures défenses, les équipes reléguées (Lorient, Clermont, Metz) se retrouvent en bas.

### 5.3 Calibration

Log-loss moyen sur les 3 types de résultats : ~0.595. Les courbes de calibration confirment que le modèle est fiable — quand il prédit 60%, la fréquence observée est proche de 60%.

### 5.4 Value bets et backtesting

| Indicateur | Valeur |
|-----------|--------|
| Seuil d'edge retenu | > 1.05 (modèle dépasse Bet365 de +5%) |
| Value bets détectés | 363 sur 306 matchs |
| Taux de réussite global | 29.8% (normal — on parie sur des outsiders) |
| EV moyen par pari | +0.165 |
| ROI mise fixe | -2.3% |
| ROI 25% Kelly | -0.1% |
| ROI 10% Kelly | +7.2% |
| Drawdown maximum | -32.2% |

---

## 6. Choix techniques et décisions de conception

### 6.1 Ce qui a été retiré de la version initiale

La version initiale du notebook contenait plusieurs éléments volontairement simplifiés pour produire un code plus naturel.

**Commentaires décoratifs supprimés**

La version initiale utilisait des séparateurs visuels lourds :

```python
# ════════════════════════════════════════════════
# INTENSITÉS DE SCORE (log-linéaire)
# ════════════════════════════════════════════════
```

Ces blocs ont été remplacés par des commentaires naturels ou supprimés quand le code se lisait de lui-même.

**Gestion d'erreurs systématique retirée**

La cellule de chargement utilisait un `try/except` pour chaque saison, et la boucle de calcul des probabilités vérifiait `if ht not in team_to_idx` pour chaque match. Ces gardes ont été retirées car elles masquaient les erreurs réelles et rendaient le code défensivement artificiel.

**Prints informatifs inutiles supprimés**

La version initiale affichait avec emojis : versions de toutes les librairies, liste des équipes avec indices, séparateurs `===` entre sections. Réduits à ce qui est réellement utile.

**Noms de variables simplifiés**

Des noms verbeux comme `summary_global`, `stakes_fixed`, `stakes_kelly` ont été remplacés ou les variables inutilisées abandonnées (les stakes du backtesting ne sont pas utilisées).

**`pm.model_to_graphviz()` commenté**

Cette ligne déclenchait le bug PyTensor sur Python 3.12.

**Docstrings allégées**

La fonction `simulate_bankroll` avait une docstring complète avec bloc `Parameters` formaté NumPy. Supprimée — le code est suffisamment lisible sans.

### 6.2 Correction du bug Python 3.12

PyTensor déclenche `NotImplementedError: float16` lors de la compilation C sur Python 3.12. Solution : désactiver le compilateur C avant tout import PyMC.

```python
import os
os.environ["PYTENSOR_FLAGS"] = "floatX=float64,optimizer=fast_compile,cxx="
```

Le paramètre `cxx=` vide désactive la compilation C — PyTensor utilise NumPy à la place. Légèrement plus lent mais fonctionne sans erreur.

### 6.3 Choix du seuil d'edge

Le seuil `edge > 1.05` est conservateur. Un seuil plus élevé (ex: 1.10) donnerait moins de paris mais potentiellement plus fiables. À 1.05, 363 value bets détectés sur 306 matchs — soit plus d'un par match, ce qui confirme le data leakage.

### 6.4 Limitation principale : data leakage

Le modèle est entraîné sur la saison 2023-24, puis évalué sur ces mêmes matchs pour les value bets. Cela gonfle artificiellement le nombre de value bets détectés et les ROI. En production, il faudrait une validation **walk-forward** : entraîner jusqu'au match j, prédire j+1.

---

## 7. Modèle dynamique (bonus)

En complément du modèle statique, un modèle dynamique pondère les matchs récents plus fortement :

```
poids(match g) = exp(lambda × (date_g - date_max))
lambda = 0.007  =>  match vieux de 100 matchs => poids 0.5
```

Ce modèle utilise les 686 matchs des deux saisons et génère une carte de forme récente par équipe.

**Limite** : la pondération via `pm.Potential()` est une approximation — ce n'est pas un vrai modèle d'état. Une approche state-space avec Random Walk gaussien sur `att` et `def` serait plus rigoureuse.

---

## 8. Extensions possibles

- **Walk-forward validation** : entraîner jusqu'au match j, prédire j+1 — élimine le data leakage
- **Correction Dixon-Coles** : ajustement pour les scores 0-0 et 1-0 sous-estimés par Poisson indépendants
- **Cotes Pinnacle** : utiliser PSH/PSD/PSA (marge ~2%) au lieu de Bet365 pour un meilleur benchmark
- **Expected Goals (xG)** : intégrer les xG comme features supplémentaires
- **Poisson bivarié** : modéliser la corrélation entre buts domicile/extérieur (Karlis & Ntzoufras, 2003)
- **State-space model** : Random Walk gaussien sur `att`/`def` pour un vrai modèle dynamique

---

## 9. Références

- Baio, G. & Blangiardo, M. (2010). *Bayesian hierarchical model for the prediction of football results.* Journal of Applied Statistics.
- Dixon, M.J. & Coles, S.G. (1997). *Modelling association football scores and inefficiencies in the football betting market.* Applied Statistics.
- Karlis, D. & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate Poisson models.* The Statistician.
- Kelly, J.L. (1956). *A New Interpretation of Information Rate.* Bell System Technical Journal.
- Données : [football-data.co.uk](https://www.football-data.co.uk) — Ligue 1 2022-23 et 2023-24
- Librairies : PyMC 5.x, ArviZ, Pandas, Matplotlib, Seaborn, SciPy
