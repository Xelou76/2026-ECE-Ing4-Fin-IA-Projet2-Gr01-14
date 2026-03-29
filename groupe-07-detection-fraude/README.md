# 2026 - ECE - Ing4 - Fin - IA Probabiliste, Theorie des Jeux et Machine Learning - Groupe 01

Projet pedagogique d'exploration des approches d'intelligence artificielle probabiliste, de la theorie des jeux et du machine learning appliques a la finance et au-dela, pour les etudiants de l'ECE Paris.

Ce depot contient **52 sujets** repartis en **9 categories** (A-I), couvrant l'inference bayesienne, la theorie des jeux, le ML quantitatif, la confidentialite, le deep learning SOTA, les agents LLM, la recherche, le trading algorithmique (QuantConnect) et des applications transversales non-finance. Chaque sujet s'appuie sur les notebooks du cours [CoursIA](https://github.com/jsboige/CoursIA) et propose des objectifs gradues (Minimum / Bon / Excellent) pour guider votre ambition.

---

## Modalites du projet

### Echeances importantes

| Etape | Date | Details |
|-------|------|---------|
| Presentation des sujets | **16 mars 2026** | Presentation en TP des sujets proposes |
| Checkpoint intermediaire | **23 mars 2026** | CM ML - point d'avancement tous groupes |
| Deadline PR | **28 mars 2026** | Pull Request sur ce depot (2 jours avant soutenance) |
| Soutenance finale | **30 mars 2026** | Presentation finale et rendu |

### Taille des groupes

La taille standard d'un groupe est de **3 personnes**.
- Groupes de 2 : tolere (+1 point bonus potentiel pour la charge)
- Groupes de 4 : tolere (-1 point malus potentiel pour la dilution)
- Individuel : exceptionnel (+3 points bonus potentiel)

### Evaluation collegiale

L'evaluation se fait par les pairs (etudiants + encadrants) sur 4 criteres notes de 0 a 10 :

| Critere | Description |
|---------|-------------|
| **Qualite de la presentation** | Communication, clarte, pedagogie, qualite des slides, demos |
| **Qualite theorique** | Principes utilises, classe d'algorithmes, contexte, explications des performances et des problemes, historique |
| **Qualite technique** | Livrables, commits, qualite du code, demos, resultats, perspectives |
| **Organisation** | Planning, repartition des taches, collaboration, documentation, integration au projet GitHub |

### Livrables attendus

- **Code source** propre et documente
- **README** complet (contexte, installation, usage, resultats)
- **Slides** de la presentation (PDF ou lien)

### Instructions de soumission

#### Organisation du travail

> **ATTENTION** : Tout votre travail **DOIT** etre organise dans un **sous-repertoire dedie** a votre groupe.
>
> **Structure obligatoire** :
> ```
> /groupe-XX-nom-sujet/
> |-- README.md          # Documentation de votre projet
> |-- src/               # Code source
> |-- docs/              # Documentation technique
> |-- slides/            # Support de presentation (PDF ou lien)
> |-- ...
> ```
>
> Ne pas mettre vos fichiers a la racine du depot.

#### Soumission du code et de la documentation

1. **Creer un fork** de ce depot sur votre compte GitHub
2. **Creer un sous-repertoire** pour votre groupe : `groupe-XX-nom-sujet/` (ex: `groupe-03-portfolio-bayesien/`)
3. **Developper votre projet** exclusivement dans ce sous-repertoire
4. **Soumettre une Pull Request** vers ce depot **au moins 2 jours avant la presentation** (soit le **28 mars 2026** au plus tard)
5. La PR doit inclure :
   - Le code source complet et fonctionnel dans votre sous-repertoire
   - Un README detaille (installation, utilisation, tests)
   - La documentation technique

#### Soumission du support de presentation

- Les slides doivent etre soumises **avant le debut de la presentation** (soit le **30 mars 2026** au matin)
- Format accepte : PDF, PowerPoint, ou lien vers Google Slides/Canva
- Ajouter les slides dans votre sous-repertoire (`groupe-XX/slides/`) ou partager le lien dans le README

#### Checklist de soumission

- [ ] Fork du depot cree
- [ ] Sous-repertoire `groupe-XX-nom-sujet/` cree avec tout le contenu dedans
- [ ] README avec procedure d'installation et tests dans le sous-repertoire
- [ ] Pull Request creee et reviewable
- [ ] Slides de presentation soumises
- [ ] Tous les membres du groupe identifies dans la PR (noms + GitHub usernames)

---

## Index des sujets

Vous etes libres de choisir l'un des sujets ci-dessous ou de proposer un sujet personnel (a faire valider par les encadrants).
**Technologie libre** : Python (recommande pour l'ecosysteme ML), C#/.NET, C++, Julia, etc.

**Navigation rapide par categorie** :
[A - Probas](#categorie-a--ia-probabiliste-et-inference-bayesienne) | [B - GameTheory](#categorie-b--theorie-des-jeux-et-systemes-multi-agents) | [C - ML Finance](#categorie-c--machine-learning-et-finance-quantitative) | [D - Privacy](#categorie-d--confidentialite-et-securite-du-ml) | [E - Deep Learning](#categorie-e--deep-learning-et-architectures-modernes) | [F - LLM Agents](#categorie-f--ia-agents-et-llms-pour-la-finance) | [G - Recherche](#categorie-g--recherche-et-innovation) | [H - QuantConnect](#categorie-h--trading-algorithmique-avec-quantconnect) | [I - Transversaux](#categorie-i--sujets-non-finance-et-transversaux)

| Cat. | # | Sujet | Difficulte | Themes |
|------|---|-------|-----------|--------|
| A | A.1 | [Recommandation Bayesienne d'Actifs Financiers](#a1---recommandation-bayesienne-dactifs-financiers) | 3/5 | Probas |
| A | A.2 | [Allocation sous Incertitude - Processus Gaussiens](#a2---allocation-dactifs-sous-incertitude-avec-processus-gaussiens) | 3/5 | Probas, ML |
| A | A.3 | [Prediction de Defaut - Modeles Hierarchiques](#a3---prediction-de-defaut-dentreprise-avec-modeles-hierarchiques) | 3/5 | Probas |
| A | A.4 | [Volatilite Stochastique (Heston/SABR) avec MCMC](#a4---modelisation-de-volatilite-stochastique-hestonsabr-avec-mcmc) | 4/5 | Probas |
| A | A.5 | [Conformal Prediction pour Risk Management](#a5---conformal-prediction-pour-risk-management) | 3/5 | Probas, ML |
| A | A.6 | [Bayesian Neural Networks pour Portefeuille](#a6---bayesian-neural-networks-pour-portefeuille) | 3/5 | Probas, ML |
| A | A.7 | [Marketing Mix Modeling Bayesien](#a7---marketing-mix-modeling-bayesien) | 3/5 | Probas |
| B | B.1 | [Auction Design pour Marches Financiers](#b1---auction-design-pour-marches-financiers) | 3/5 | GameTheory |
| B | B.2 | [Jeux d'Investissement sur Graphes (Network Games)](#b2---jeux-dinvestissement-sur-graphes-network-games) | 3/5 | GameTheory |
| B | B.3 | [Negociation Automatique (CFR) en Finance](#b3---negociation-automatique-cfr-en-finance) | 4/5 | GameTheory |
| B | B.4 | [Formation de Coalitions - Trading Cooperatif](#b4---formation-de-coalitions-pour-plateformes-de-trading) | 3/5 | GameTheory |
| B | B.5 | [Dynamique Evolutionniste de Strategies de Trading](#b5---dynamique-evolutionniste-de-strategies-de-trading) | 3/5 | GameTheory |
| B | B.6 | [Mean Field Games pour Dynamique de Marche](#b6---mean-field-games-pour-dynamique-de-marche) | 4/5 | GameTheory, ML |
| C | C.1 | [Classification Documents Financiers Zero-Shot](#c1---classification-de-documents-financiers-avec-zero-shot-learning) | 3/5 | ML, NLP |
| C | C.2 | [RAG pour Questions Financieres Complexes](#c2---rag-pour-questions-financieres-complexes) | 3/5 | ML, NLP |
| C | C.3 | [Detection de Regimes de Marche (VAE-HMM)](#c3---detection-de-regimes-de-marche-vae-hmm) | 4/5 | ML, Probas |
| C | C.4 | [Classification Risque ESG Multi-label](#c4---classification-risque-esg-multi-label) | 3/5 | ML, NLP |
| C | C.5 | [Optimisation de Portefeuille Bayesien (Black-Litterman)](#c5---optimisation-de-portefeuille-bayesien-black-litterman) | 3/5 | Probas, ML |
| C | C.6 | [Credit Scoring avec IA Explicable (XAI)](#c6---credit-scoring-avec-ia-explicable-xai) | 3/5 | ML |
| C | C.7 | [Detection de Fraude en Temps Reel](#c7---detection-de-fraude-en-temps-reel) | 3/5 | ML |
| D | D.1 | [Federated Learning pour Credit Collaboratif](#d1---federated-learning-pour-prediction-de-defaut-collaborative) | 3/5 | ML, Privacy |
| D | D.2 | [Chiffrement Homomorphe (FHE) pour Finance](#d2---chiffrement-homomorphe-fhe-pour-agregation-de-donnees-financieres) | 4/5 | Crypto, ML |
| D | D.3 | [Detection de Data Poisoning Adversarial](#d3---detection-de-data-poisoning-adversarial) | 3/5 | ML, Securite |
| E | E.1 | [Foundation Models pour Series Financieres (Kronos)](#e1---foundation-models-pour-series-financieres-kronos) | 3/5 | ML, Probas |
| E | E.2 | [Transformers pour Limit Order Book (TLOB)](#e2---transformers-pour-limit-order-book-tlob) | 4/5 | ML |
| E | E.3 | [Diffusion Models pour Donnees Financieres Synthetiques](#e3---diffusion-models-pour-donnees-financieres-synthetiques) | 4/5 | ML, Probas |
| E | E.4 | [Mamba/SSM pour Prediction Financiere](#e4---mambassm-pour-prediction-financiere) | 3/5 | ML |
| E | E.5 | [GNN pour Construction de Portefeuille](#e5---gnn-pour-construction-de-portefeuille) | 4/5 | ML, Probas |
| E | E.6 | [PINNs pour Pricing d'Options](#e6---pinns-pour-pricing-doptions) | 4/5 | Probas, ML |
| F | F.1 | [Multi-Agent LLM Trading (TradingAgents)](#f1---multi-agent-llm-trading-tradingagents) | 4/5 | GameTheory, ML |
| F | F.2 | [LLM Sentiment Alpha Generation (DK-CoT)](#f2---llm-sentiment-alpha-generation-dk-cot) | 3/5 | ML, NLP |
| F | F.3 | [FinGPT Fine-Tuning pour Taches Financieres](#f3---fingpt-fine-tuning-pour-taches-financieres) | 3/5 | ML, NLP |
| F | F.4 | [LLMs pour Generation de Scenarios Macro](#f4---llms-pour-generation-de-scenarios-macroeconomiques) | 3/5 | ML |
| F | F.5 | [Neurosymbolic AI pour Decisions de Credit](#f5---neurosymbolic-ai-pour-decisions-de-credit) | 4/5 | ML, SymbolicAI |
| G | G.1 | [Causal ML pour Asset Pricing (EconML/DoWhy)](#g1---causal-ml-pour-asset-pricing-econmldowhy) | 3/5 | Probas, ML |
| G | G.2 | [GNN Risque Systemique et Contagion](#g2---gnn-risque-systemique-et-contagion-financiere) | 4/5 | GameTheory, ML |
| G | G.3 | [RL Market Making et Execution Optimale](#g3---rl-market-making-et-execution-optimale) | 4/5 | GameTheory, ML |
| G | G.4 | [World Models pour Trading (DreamerV3)](#g4---world-models-pour-trading-dreamerv3) | 4/5 | GameTheory, ML |
| G | G.5 | [Imitation-RL pour Controle Stochastique (FinFlowRL)](#g5---imitation-rl-pour-controle-stochastique-en-finance-finflowrl) | 5/5 | Probas, GT, ML |
| G | G.6 | [GFlowNets pour Generation de Portefeuilles](#g6---gflownets-pour-generation-de-portefeuilles) | 4/5 | Probas, ML |
| H | H.1 | [Strategie Alpha ML sur QuantConnect](#h1---strategie-alpha-ml-sur-quantconnect) | 3/5 | ML, QC |
| H | H.2 | [Deep RL Trading avec QuantConnect](#h2---deep-rl-trading-avec-quantconnect) | 4/5 | ML, GT, QC |
| H | H.3 | [Composite AlphaModel Framework](#h3---composite-alphamodel-framework) | 3/5 | ML, QC |
| H | H.4 | [Regime Switching et Allocation Adaptative](#h4---regime-switching-et-allocation-adaptative) | 3/5 | ML, Probas, QC |
| H | H.5 | [Options Strategies Automatisees](#h5---options-strategies-automatisees-wheelcovered-call) | 3/5 | ML, QC |
| H | H.6 | [Walk-Forward Analysis et Robustesse](#h6---walk-forward-analysis-et-robustesse-de-strategies) | 3/5 | ML, QC |
| I | I.1 | [TrueSkill et Matchmaking Competitif](#i1---trueskill-et-matchmaking-competitif) | 3/5 | Probas |
| I | I.2 | [Bayesian Sports Analytics](#i2---bayesian-sports-analytics) | 3/5 | Probas |
| I | I.3 | [Hanabi AI - Cooperation et Theory of Mind](#i3---hanabi-ai---cooperation-et-theory-of-mind) | 4/5 | GameTheory |
| I | I.4 | [Rational Speech Acts (RSA) - Pragmatique du Langage](#i4---rational-speech-acts-rsa---pragmatique-du-langage) | 3/5 | Probas |
| I | I.5 | [Kidney Exchange - Optimisation Combinatoire Cooperative](#i5---kidney-exchange---optimisation-combinatoire-cooperative) | 4/5 | GameTheory |
| I | I.6 | [RL pour Controle de Jeux (Snake/Mario/CartPole)](#i6---rl-pour-controle-de-jeux-snakemariocartpole) | 3/5 | ML |

---

## Categorie A : IA Probabiliste et Inference Bayesienne

Ces sujets explorent l'incertitude, l'inference bayesienne et la modelisation statistique appliquees a la finance. Ils demandent une bonne comprehension des distributions de probabilites, des graphes de facteurs et de la programmation probabiliste.

---

### A.1 - Recommandation Bayesienne d'Actifs Financiers

**Difficulte** : 3/5 | **Domaine** : Probas

**Description** :
Construire un systeme de recommandation bayesien qui suggere des actifs financiers aux investisseurs en se basant sur leur profil de risque et les performances historiques. Le systeme utilise l'inference probabiliste pour modeliser les preferences latentes des investisseurs et les caracteristiques des actifs, puis met a jour ses croyances en ligne avec les nouvelles donnees de marche. Contrairement aux approches de filtrage collaboratif classique, l'approche bayesienne permet de quantifier l'incertitude sur chaque recommandation.

**Objectifs gradues** :
- **Minimum** : Modele bayesien simple avec Infer.NET, profil de risque statique, recommandation basee sur rendement moyen
- **Bon** : Inference en ligne avec mises a jour dynamiques, incorporation de mesures de risque (VaR, CVaR), visualisation des distributions posterieures
- **Excellent** : Modele de melange gaussien pour clustering d'actifs, apprentissage des preferences utilisateur, backtesting historique sur donnees reelles, interface utilisateur

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Introduction Infer.NET - modeles probabilistes | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |
| Infer-12 | Systemes de recommandation probabilistes | [Infer-12](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-12-Recommenders.ipynb) |

**References externes** :
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) - Optimisation de portefeuille en Python
- [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) - Bibliotheque avancee d'optimisation de portefeuille
- [Infer.NET Documentation](https://dotnet.github.io/infer/) - Framework de programmation probabiliste Microsoft

---

### A.2 - Allocation d'Actifs sous Incertitude avec Processus Gaussiens

**Difficulte** : 3/5 | **Domaine** : Probas, ML

**Description** :
Utiliser des processus gaussiens (GP) pour modeliser l'incertitude sur les rendements futurs et optimiser l'allocation d'actifs en consequence. Les GP fournissent une estimation non-parametrique avec bandes de confiance "gratuites", particulierement adaptees aux donnees financieres ou les regimes changent et ou l'incertitude est fondamentale. Le systeme doit integrer cette incertitude dans la decision d'investissement pour construire des portefeuilles robustes.

**Objectifs gradues** :
- **Minimum** : Processus gaussien univarie pour modeliser un actif, optimisation portefeuille simple (Sharpe ratio)
- **Bon** : GP multivarie pour correlations entre actifs, optimisation sous contraintes (budget, exposition sectorielle), visualisation des bandes de confiance
- **Excellent** : GP avec noyaux adaptatifs pour plusieurs actifs, optimisation robuste (CVaR), comparaison avec methodes classiques (Markowitz), backtesting sur donnees reelles

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Introduction Infer.NET - inference bayesienne | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

**References externes** :
- [GPyTorch](https://gpytorch.ai/) - Processus gaussiens scalables avec PyTorch (point de depart recommande)
- [GPyTorch Regression Tutorial](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html) - Tutoriel pas a pas
- [Deep Kernel Learning (arXiv)](https://arxiv.org/abs/1511.02222) - Combinaison GP + deep learning
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) - Optimisation de portefeuille
- [Scikit-learn GP](https://scikit-learn.org/stable/modules/gaussian_process.html) - Implementation de base

---

### A.3 - Prediction de Defaut d'Entreprise avec Modeles Hierarchiques

**Difficulte** : 3/5 | **Domaine** : Probas

**Description** :
Modele probabiliste pour predire la probabilite de defaut d'entreprises cotees en combinant donnees financieres (ratios, bilans) et macroeconomiques (taux, inflation). L'approche hierarchique permet de partager l'information entre secteurs d'activite tout en capturant les specificites de chacun. Le modele fournit des predictions avec intervalles de confiance, essentiels pour les decisions de credit.

**Objectifs gradues** :
- **Minimum** : Regression logistique bayesienne avec Infer.NET, predictions ponctuelles sur donnees synthethiques
- **Bon** : Modele hierarchique par secteurs d'activite, incorporation de variables macro, calibration des probabilites
- **Excellent** : Modele de survie bayesien (temps jusqu'au defaut), comparaison avec approches frequentistes, backtesting sur donnees historiques

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Introduction Infer.NET - modeles probabilistes | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

**References externes** :
- [PyMC Hierarchical Regression Tutorial](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-hierarchical.html) - Modele hierarchique de reference (point de depart recommande)
- [Kaggle - Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) - Donnees de prets reelles
- [Infer.NET Bayesian Logistic Regression](https://dotnet.github.io/infer/userguide/Logistic%20regression.html) - Tutoriel officiel
- [Survival Analysis in Python (lifelines)](https://lifelines.readthedocs.io/) - Pour le niveau Excellent

---

### A.4 - Modelisation de Volatilite Stochastique (Heston/SABR) avec MCMC

**Difficulte** : 4/5 | **Domaine** : Probas

**Description** :
Implementer un modele de volatilite stochastique (Heston ou SABR) en utilisant la programmation probabiliste (Pyro) et l'inference MCMC. La volatilite des actifs financiers n'est pas constante : elle varie dans le temps avec des clusters de haute volatilite. Les modeles stochastiques capturent cette dynamique latente. Ce sujet connecte directement la programmation probabiliste moderne et la finance quantitative avancee (pricing d'options, risk management).

**Objectifs gradues** :
- **Minimum** : Modele Heston simple avec Pyro, inference MCMC basique, estimation sur donnees synthetiques
- **Bon** : Modele SABR (Stochastic Alpha Beta Rho), diagnostics de convergence MCMC (R-hat, ESS), comparaison formule fermee vs MCMC
- **Excellent** : Calibration sur donnees de marche d'options reelles, vol surface fitting, pricing d'options exotiques, visualisation des chaines MCMC

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Pyro_RSA | Programmation probabiliste avec Pyro | [Pyro_RSA](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Pyro_RSA_Hyperbole.ipynb) |
| Infer-101 | Introduction inference bayesienne | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

**References externes** :
- [Pyro - Probabilistic Programming](https://pyro.ai/) - Framework PPL de Uber AI
- [Pyro SVI Tutorial](https://pyro.ai/examples/svi_part_i.html) - Tutoriel inference variationnelle (point de depart)
- [NumPyro](https://num.pyro.ai/) - Version JAX de Pyro (plus rapide pour MCMC)
- [NumPyro Stochastic Volatility Example](https://num.pyro.ai/en/stable/examples/stochastic_volatility.html) - Implementation directe du sujet
- [Heston Model Wikipedia](https://en.wikipedia.org/wiki/Heston_model) - Reference theorique

---

### A.5 - Conformal Prediction pour Risk Management

**Difficulte** : 3/5 | **Domaine** : Probas, ML

**Description** :
Appliquer la conformal prediction -- un cadre de quantification d'incertitude distribution-free -- pour fournir des intervalles de prediction avec couverture garantie pour les previsions financieres (VaR, prix d'options, rendements de portefeuille). Contrairement aux methodes bayesiennes qui necessitent des hypotheses sur la distribution, la conformal prediction garantit mathematiquement le taux de couverture quel que soit le modele sous-jacent.

Les travaux recents (CPPS 2024, ACI pour crypto 2024, NeurIPS 2025) montrent l'applicabilite directe a la construction de portefeuilles et au risk management avec garanties formelles.

**Objectifs gradues** :
- **Minimum** : Conformal prediction basique (split conformal) pour regression de rendements, intervalles avec couverture 95%
- **Bon** : Adaptive Conformal Inference (ACI) pour series temporelles, evaluation dynamique de la couverture, visualisation des intervalles
- **Excellent** : Application au portefeuille (CPPS), comparaison avec methodes bayesiennes et quantile regression, evaluation sur periodes de crise

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Inference bayesienne et quantification d'incertitude | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

> **Note** : Ce sujet est principalement base sur des tutoriels Python externes (MAPIE, crepes). Les notebooks du cours fournissent le contexte bayesien.

**References externes** :
- [MAPIE (Python)](https://github.com/scikit-learn-contrib/MAPIE) - Bibliotheque de conformal prediction pour scikit-learn (point de depart recommande)
- [MAPIE Tutorials](https://mapie.readthedocs.io/en/latest/examples_regression/index.html) - Tutoriels pas a pas
- [Awesome Conformal Prediction](https://github.com/valeman/awesome-conformal-prediction) - Liste curee de ressources
- [CPPS - Conformal Predictive Portfolio Selection (2024)](https://arxiv.org/pdf/2410.16333) - Application portefeuille
- [ACI for Crypto VaR (2024)](https://www.mdpi.com/1911-8074/17/6/248) - Conformal prediction adaptative
- [Conformal Stock Selection (MLR 2025)](https://proceedings.mlr.press/v266/kaya25a.html) - Selection d'actions avec garanties

---

### A.6 - Bayesian Neural Networks pour Portefeuille

**Difficulte** : 3/5 | **Domaine** : Probas, ML

**Description** :
Implementer des reseaux de neurones bayesiens (BNN) ou chaque poids est une distribution de probabilite, permettant au modele de dire "je ne sais pas" (incertitude epistemique). Utiliser les distributions posterieures predictives pour construire un portefeuille qui penalise l'incertitude : allouer plus aux actifs ou le modele est confiant, moins a ceux ou l'incertitude est elevee.

Ce sujet fait le pont entre deep learning et probabilites bayesiennes, avec une application directe a l'allocation d'actifs risk-aware.

**Objectifs gradues** :
- **Minimum** : BNN simple avec NumPyro ou PyMC pour predire les rendements de quelques actifs, extraction des posterieures
- **Bon** : Portefeuille mean-variance avec covariance bayesienne, comparaison BNN vs point-estimate, visualisation de l'incertitude
- **Excellent** : Analyse du comportement en periode de crise (incertitude augmente-t-elle ?), comparaison avec ensemble methods, portfolio risk-adjusted

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Pyro_RSA | Programmation probabiliste avec Pyro | [Pyro_RSA](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Pyro_RSA_Hyperbole.ipynb) |
| Infer-101 | Introduction inference bayesienne | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

**References externes** :
- [NumPyro BNN Example](https://github.com/pyro-ppl/numpyro/blob/master/examples/bnn.py) - Implementation de reference
- [PyMC BNN Tutorial](https://www.pymc.io/projects/examples/en/latest/variational_inference/bayesian_neural_network_advi.html) - Tutoriel ADVI
- [TyXe (Pyro BNNs)](https://github.com/cifkao/tyxe) - Bibliotheque BNN sur Pyro
- [Pyro BNN Tutorial](https://pyro.ai/examples/bnn.html) - Tutoriel officiel

---

### A.7 - Marketing Mix Modeling Bayesien

**Difficulte** : 3/5 | **Domaine** : Probas

**Description** :
Un sujet tres demande en entreprise : optimiser le budget publicitaire. Le Marketing Mix Modeling (MMM) attribue les ventes aux differents canaux (TV, Facebook, Google) en tenant compte des effets de saturation (rendements decroissants) et de delai temporel (Adstock). L'approche bayesienne avec PyMC permet d'estimer ces parametres inconnus avec quantification d'incertitude, et de simuler des scenarios d'allocation optimale.

**Objectifs gradues** :
- **Minimum** : Modele lineaire bayesien simple avec PyMC, 2-3 canaux, estimation des coefficients
- **Bon** : Modele hierarchique avec effets de saturation (Hill transform) et Adstock, optimisation budget, visualisation des contributions
- **Excellent** : Modele multi-marche hierarchique, validation croisee temporelle, comparaison avec Google LightweightMMM, simulation de scenarios

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Introduction inference bayesienne | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

> **Note** : Ce sujet utilise principalement PyMC (Python). Le notebook Infer-101 fournit les bases bayesiennes ; les tutoriels PyMC-Marketing sont le vrai point de depart.

**References externes** :
- [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) - MMM bayesien avec PyMC (point de depart recommande)
- [PyMC-Marketing MMM Tutorial](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_example.html) - Tutoriel complet pas a pas
- [Google LightweightMMM](https://github.com/google/lightweight_mmm) - Implementation Google (pour comparaison)
- [Bayesian Methods for Media Mix Modeling (Jin et al.)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf) - Paper de reference

[Retour a l'index des sujets](#index-des-sujets)

---

## Categorie B : Theorie des Jeux et Systemes Multi-Agents

Ces sujets traitent de la prise de decision strategique, de la cooperation et de la competition entre agents autonomes. Ils s'appuient sur les notebooks GameTheory du cours et explorent des applications financieres de la theorie des jeux.

---

### B.1 - Auction Design pour Marches Financiers

**Difficulte** : 3/5 | **Domaine** : GameTheory

**Description** :
Concevoir et implementer un mecanisme d'enchere pour marches financiers (VCG, GSP ou variantes). Le systeme simule des agents strategiques avec des valorisations privees et analyse les proprietes du mecanisme : veridique (truthfulness), efficience allocative et revenu. Application directe du mechanism design a un cas d'usage reel (publicite programmatique, marches d'electricite, IPO).

**Objectifs gradues** :
- **Minimum** : Enchere au second prix (Vickrey) pour 1 slot, simulation d'annonceurs avec valeurs privees
- **Bon** : Enchere GSP (Generalized Second Price) pour plusieurs slots, calcul de l'equilibre de Nash
- **Excellent** : Comparaison VCG vs GSP, analyse des incitations a tricher, optimisation du revenu, extension au budgeting dynamique

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-16 | Mechanism Design | [GT-16](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-16-MechanismDesign.ipynb) |
| GT-15 | Jeux cooperatifs | [GT-15](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-15-CooperativeGames.ipynb) |
| GT-11 | Jeux bayesiens | [GT-11](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-11-BayesianGames.ipynb) |

**References externes** :
- [Optimal Auctions through Deep Learning (arXiv)](https://arxiv.org/abs/1905.05533) - RegretNet
- [GitHub RegretNet](https://github.com/srp3/regretnet) - Implementation de reference
- [OpenSpiel (DeepMind)](https://github.com/deepmind/open_spiel) - Framework de jeux

---

### B.2 - Jeux d'Investissement sur Graphes (Network Games)

**Difficulte** : 3/5 | **Domaine** : GameTheory

**Description** :
Modeliser des decisions d'investissement dans un reseau ou les rendements dependent des investissements des voisins (effets de reseau, externalites). Les joueurs sont les noeuds d'un graphe et leurs gains dependent de leurs propres actions et de celles de leurs voisins. Ce modele capture les phenomenes de contagion financiere, d'effets de mode et de cascades informationnelles.

**Objectifs gradues** :
- **Minimum** : Graphe simple, jeu de contribution binaire, calcul d'un equilibre de Nash
- **Bon** : Graphes complexes (scale-free, small-world), dynamique d'apprentissage (fictitious play), visualisation
- **Excellent** : Information incomplete, analyse de stabilite, comparaison avec graphes reels (reseaux d'investisseurs), optimal network design

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-4 | Equilibre de Nash | [GT-4](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-4-NashEquilibrium.ipynb) |
| GT-6 | Jeux evolutionnistes et confiance | [GT-6](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-6-EvolutionTrust.ipynb) |

**References externes** :
- [NetworkX](https://networkx.org/) - Bibliotheque Python pour graphes
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html) - Prise en main rapide avec exemples
- [Game Theory on Networks (Jackson, 2008)](https://web.stanford.edu/~jacksonm/netbook.pdf) - Reference theorique
- [Bramoull et al. (2014)](https://www.sciencedirect.com/science/article/pii/S0014292114000415) - Survey "Strategic Interaction and Networks" (European Economic Review)
- [OpenSpiel (DeepMind)](https://github.com/deepmind/open_spiel) - Framework de jeux (inclut network games)

---

### B.3 - Negociation Automatique (CFR) en Finance

**Difficulte** : 4/5 | **Domaine** : GameTheory

**Description** :
Implementer un agent de negociation automatique utilisant des algorithmes de resolution de jeux a information imparfaite (Counterfactual Regret Minimization - CFR). Application a un scenario de negociation financiere (achat/vente d'actif, negociation de prix). Le CFR, rendu celebre par les IA de poker (Libratus, Pluribus), converge vers un equilibre de Nash dans les jeux a somme nulle a deux joueurs.

**Objectifs gradues** :
- **Minimum** : Jeu de negociation simplifie (offre alternee), implementation CFR basique
- **Bon** : Jeu avec plus d'etats, incorporation de signaux prives, analyse de la strategie d'equilibre
- **Excellent** : Negociation multi-issues (prix + quantite + delai), apprentissage adversarial, evaluation contre humains

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-13 | Information imparfaite et CFR | [GT-13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-13-ImperfectInfo-CFR.ipynb) |
| GT-11 | Jeux bayesiens | [GT-11](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-11-BayesianGames.ipynb) |

**References externes** :
- [OpenSpiel (DeepMind)](https://github.com/deepmind/open_spiel) - Framework CFR et variantes
- [Libratus (Science, 2018)](https://science.sciencemag.org/content/359/6374/418) - IA poker superhuman
- [Pluribus (Science, 2019)](https://science.sciencemag.org/content/365/6456/885) - Poker multiplayer

---

### B.4 - Formation de Coalitions pour Plateformes de Trading

**Difficulte** : 3/5 | **Domaine** : GameTheory

**Description** :
Modeliser la formation de coalitions entre traders sur une plateforme d'echange pour reduire les frais de transaction ou acceder a de meilleures liquidites. Analyser la stabilite des coalitions avec les concepts de jeux cooperatifs : le core, la valeur de Shapley, le nucleolus. Ce sujet connecte les jeux cooperatifs et la microstructure de marche.

**Objectifs gradues** :
- **Minimum** : Jeu de coalition simple (frais de transaction reductibles en groupe), calcul de la valeur de Shapley
- **Bon** : Algorithme de formation de coalitions (merge-and-split), analyse de stabilite du core, visualisation
- **Excellent** : Extension au marche avec ordres limites, mecanisme d'incitation, simulation dynamique, comparaison avec plateformes reelles

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-15 | Jeux cooperatifs (Shapley, core) | [GT-15](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-15-CooperativeGames.ipynb) |
| GT-16 | Mechanism Design | [GT-16](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-16-MechanismDesign.ipynb) |

**References externes** :
- [Cooperative Game Theory (Chalkiadakis et al.)](https://www.morganclaypool.com/doi/abs/10.2200/S00355ED1V01Y201111AIM016) - Reference
- [OpenSpiel](https://github.com/deepmind/open_spiel) - Framework multi-agents

---

### B.5 - Dynamique Evolutionniste de Strategies de Trading

**Difficulte** : 3/5 | **Domaine** : GameTheory

**Description** :
Simuler une population de strategies de trading (trend-following, mean-reversion, buy-and-hold, noise trading) qui evoluent par selection naturelle : les strategies profitables se propagent, les perdantes disparaissent. Analyser les dynamiques de type replicator et identifier les Evolutionarily Stable Strategies (ESS). Ce modele eclaire pourquoi certaines strategies marchent puis cessent de fonctionner (crowding effect).

**Objectifs gradues** :
- **Minimum** : Population de 2-3 strategies simples, dynamique de replication, visualisation de l'evolution
- **Bon** : Plusieurs strategies avec parametres evolutifs, paysage de fitness complexe, analyse ESS
- **Excellent** : Coevolution de strategies, invasion mutante, backtesting sur donnees reelles, comparaison avec la finance comportementale

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-6 | Jeux evolutionnistes | [GT-6](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-6-EvolutionTrust.ipynb) |
| GT-17 | Multi-Agent RL | [GT-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL.ipynb) |

**References externes** :
- [EGTtools](https://github.com/Socrats/EGTtools) - Evolutionary Game Theory en Python
- [EGTtools Tutorials](https://egttools.readthedocs.io/en/stable/tutorials.html) - Notebooks d'exemples (replicator, Moran process)
- [Evolutionary Dynamics (Nowak, 2006)](https://www.hup.harvard.edu/books/9780674023383) - Reference theorique
- [Axelrod Python](https://github.com/Axelrod-Python/Axelrod) - Tournois de strategies (Prisoner's Dilemma, cooperation)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL pour comparaison avec approches evolutionnistes

---

### B.6 - Mean Field Games pour Dynamique de Marche

**Difficulte** : 4/5 | **Domaine** : GameTheory, ML

**Description** :
Comment modeliser l'interaction strategique d'une foule immense de traders sur un marche ? Les Mean Field Games (MFG) resolvent ce probleme en modelisant un agent representatif face a une "distribution moyenne" des autres, plutot que N agents individuels. L'approche utilise des Neural ODEs pour resoudre les equations differentielles stochastiques couplees (Hamilton-Jacobi-Bellman + Fokker-Planck).

**Objectifs gradues** :
- **Minimum** : MFG simple en dimension 1 avec resolution numerique (differences finies), visualisation
- **Bon** : Resolution par Neural ODE, application a un marche simplifie, analyse de l'equilibre
- **Excellent** : MFG a plusieurs populations (institutionnels vs retail), calibration sur donnees, comparaison avec MARL

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-4 | Equilibre de Nash (base theorique) | [GT-4](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-4-NashEquilibrium.ipynb) |
| GT-14 | Jeux differentiels | [GT-14](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-14-DifferentialGames.ipynb) |

**References externes** :
- [Mean Field Games & ML (arXiv)](https://arxiv.org/abs/2003.06069) - Paper fondateur
- [Tutoriel MFG (GitHub)](https://github.com/Nathan-Sanglier/M2MO-Mean-Field-Games) - Implementation pedagogique
- [Neural ODE (Chen et al., NeurIPS 2018)](https://arxiv.org/abs/1806.07366) - Architecture de base

[Retour a l'index des sujets](#index-des-sujets)

---

## Categorie C : Machine Learning et Finance Quantitative

Ces sujets combinent ML moderne (NLP, deep learning, XAI) et applications financieres concretes. Ils s'appuient sur les notebooks ML du cours et les labs Data Science with Agents.

---

### C.1 - Classification de Documents Financiers avec Zero-Shot Learning

**Difficulte** : 3/5 | **Domaine** : ML, NLP

**Description** :
Systeme de classification automatique de documents financiers (rapports annuels, prospectus, documents ESG) utilisant le zero-shot learning avec LLMs. Le systeme classe des documents sans entrainement supervise specifique, en exploitant les capacites de comprehension des LLMs. L'interet pratique est enorme : les equipes compliance et analyse traitent des milliers de documents par an.

**Objectifs gradues** :
- **Minimum** : Classification binaire simple avec un LLM (OpenAI/OpenRouter), prompt engineering basique
- **Bon** : Multi-classe, evaluation sur dataset reel, comparaison avec approche supervisee, few-shot prompting
- **Excellent** : Classification hierarchique, incorporation de schemas XBRL, evaluation metrologique (precision/recall/F1), interface de validation

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Lab2 | Analyse de documents (RFP Analysis) | [Lab2](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/PythonAgentsForDataScience/Day2/Labs/Lab2-RFP-Analysis/Lab2-RFP-Analysis.ipynb) |
| Lab13 | Recherche web et SOTA avec agents | [Lab13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/AgenticDataScience/Day6-MLE-Star/Lab13-Web-Search-SOTA.ipynb) |

**References externes** :
- [HuggingFace Transformers](https://huggingface.co/transformers/) - Modeles pre-entraines NLP
- [FinBERT](https://github.com/ProsusAI/finBERT) - BERT fine-tune pour la finance

---

### C.2 - RAG pour Questions Financieres Complexes

**Difficulte** : 3/5 | **Domaine** : ML, NLP

**Description** :
Systeme RAG (Retrieval Augmented Generation) permettant de poser des questions complexes sur des donnees et rapports financiers et d'obtenir des reponses avec citations. Le systeme gere des documents heterogenes (PDF, tableaux, news) et decompose les requetes complexes en sous-requetes. L'architecture RAG est la plus demandee en entreprise en 2025-2026 pour les applications IA domain-specific.

**Objectifs gradues** :
- **Minimum** : RAG basique avec vector store (Chroma), embedding simple, LLM pour generation
- **Bon** : Chunking intelligent pour tableaux financiers, multi-retrieval (dense + sparse), evaluation RAGAS
- **Excellent** : RAG agentic (requetes complexes decomposees), time-aware retrieval, citations verifiables, interface chat

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Lab13 | Recherche web et SOTA | [Lab13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/AgenticDataScience/Day6-MLE-Star/Lab13-Web-Search-SOTA.ipynb) |
| Lab7 | Agent d'analyse de donnees | [Lab7](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/PythonAgentsForDataScience/Day3/Labs/Lab7-Data-Analysis-Agent/Lab7-Data-Analysis-Agent.ipynb) |

**References externes** :
- [LangChain](https://www.langchain.com/) - Framework RAG de reference
- [LlamaIndex](https://www.llamaindex.ai/) - Framework RAG pour documents structures
- [Chroma](https://www.trychroma.com/) - Vector store open-source
- [RAGAS](https://github.com/explodinggradients/ragas) - Evaluation de systemes RAG

---

### C.3 - Detection de Regimes de Marche (VAE-HMM)

**Difficulte** : 4/5 | **Domaine** : ML, Probas

**Description** :
Systeme de detection automatique de changements de regime de marche (bull/bear, haute/basse volatilite) utilisant des techniques hybrides deep learning + modeles probabilistes. Les Variational Autoencoders (VAE) apprennent des representations latentes des etats de marche, tandis que les Hidden Markov Models (HMM) modelisent les transitions entre regimes. Une strategie adaptative change d'allocation selon le regime detecte.

**Objectifs gradues** :
- **Minimum** : Detection basee sur seuils statistiques, visualisation des regimes identifies
- **Bon** : VAE pour features latentes + HMM pour transitions, evaluation sur donnees historiques
- **Excellent** : Modele hybride VAE-HMM, prediction de transitions, backtesting de strategie adaptative, comparaison avec Markov-switching classique

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Modeles probabilistes et variables latentes | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |
| Infer-11 | Modeles de sequences (HMM) | [Infer-11](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-11-Sequences.ipynb) |
| QC-Py-24 | Autoencoders et detection d'anomalies | [QC-Py-24](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-24-Autoencoders-Anomaly.ipynb) |

**References externes** :
- [hmmlearn](https://hmmlearn.readthedocs.io/) - HMM en Python (point de depart pour la detection de regimes)
- [hmmlearn Stock Market Tutorial](https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_hmm_stock_analysis.html) - Exemple applique a la finance
- [PyTorch VAE Tutorial](https://github.com/pytorch/examples/tree/main/vae) - Implementation VAE
- [Hamilton Regime Switching Model](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_regression.html) - Markov-switching classique (statsmodels)

---

### C.4 - Classification Risque ESG Multi-label

**Difficulte** : 3/5 | **Domaine** : ML, NLP

**Description** :
Systeme de classification multi-label de documents financiers selon les criteres ESG (Environnemental, Social, Gouvernance). Un meme document peut aborder plusieurs criteres simultanement. Le systeme utilise des LLMs pour extraire et classifier les mentions ESG, puis les agrege au niveau entreprise. L'ESG est un enjeu majeur en finance reglementaire (SFDR, taxonomie UE) en 2025-2026.

**Objectifs gradues** :
- **Minimum** : Classification binaire par critere ESG avec un LLM, prompt basique
- **Bon** : Classification multi-label, evaluation multi-label (precision@k, recall@k, hamming loss)
- **Excellent** : Extraction de mentions ESG au niveau citation, agregation entreprise, benchmark contre annotations humaines, dashboard ESG

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Lab3 | CV Screening (classification de textes) | [Lab3](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/PythonAgentsForDataScience/Day2/Labs/Lab3-CV-Screening/Lab3-CV-Screening.ipynb) |
| Lab13 | Recherche web et SOTA | [Lab13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/AgenticDataScience/Day6-MLE-Star/Lab13-Web-Search-SOTA.ipynb) |

**References externes** :
- [HuggingFace Transformers](https://huggingface.co/transformers/) - Modeles NLP
- [FinBERT](https://github.com/ProsusAI/finBERT) - BERT pour la finance
- [SFDR Documentation (EU)](https://finance.ec.europa.eu/sustainable-finance/disclosures/sustainability-related-disclosure-financial-services-sector_en) - Reglementation

---

### C.5 - Optimisation de Portefeuille Bayesien (Black-Litterman)

**Difficulte** : 3/5 | **Domaine** : Probas, ML

**Description** :
Au-dela de la theorie de Markowitz classique, utiliser le modele Black-Litterman pour integrer des "views" (opinions) probabilistes sur les rendements futurs. L'approche bayesienne combine un prior (equilibre de marche) avec des views de l'investisseur pour obtenir une allocation plus stable et intuitive. Ce modele est utilise par la majorite des asset managers institutionnels.

**Objectifs gradues** :
- **Minimum** : Implementation Black-Litterman avec views simples, comparaison avec Markowitz classique sur donnees Yahoo Finance
- **Bon** : Views avec niveaux de confiance variables, optimisation sous contraintes (budget, secteur), frontiere efficiente
- **Excellent** : Views generees par ML (sentiment, momentum), backtesting multi-periodes, analyse de sensibilite aux views

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Inference bayesienne (prior + posterior) | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |
| QC-Py-21 | Optimisation de portefeuille ML | [QC-Py-21](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-21-Portfolio-Optimization-ML.ipynb) |

**References externes** :
- [PyPortfolioOpt - Black-Litterman](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html) - Implementation directe en Python (point de depart recommande)
- [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) - Optimisation avancee avec Black-Litterman
- [Black-Litterman Model (Wikipedia)](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model) - Reference theorique
- [Thomas Starke - BL Model (QuantConnect Learning)](https://www.quantconnect.com/learning/introduction-to-financial-python) - Tutoriels financiers Python

---

### C.6 - Credit Scoring avec IA Explicable (XAI)

**Difficulte** : 3/5 | **Domaine** : ML

**Description** :
Le scoring de credit est un cas d'usage majeur du ML en finance, avec des contraintes reglementaires fortes : un client refuse a le droit de savoir pourquoi (RGPD, article 22). Le projet entraine un modele performant (XGBoost, LightGBM) puis applique des methodes d'explicabilite (SHAP, LIME, explications contrefactuelles) pour rendre les decisions transparentes. L'analyse des biais (age, genre) est un bonus significatif.

**Objectifs gradues** :
- **Minimum** : Modele de scoring (XGBoost) sur dataset public (German Credit, Lending Club), SHAP values
- **Bon** : Comparaison SHAP vs LIME, explications contrefactuelles ("que changer pour etre accepte ?"), analyse de biais
- **Excellent** : Dashboard interactif d'explicabilite, audit de fairness (equalized odds), comparaison modele boite noire vs interpretable

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Lab3 | CV Screening (classification) | [Lab3](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/PythonAgentsForDataScience/Day2/Labs/Lab3-CV-Screening/Lab3-CV-Screening.ipynb) |
| ML-4 | Evaluation de modeles ML | [ML-4](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-4-Evaluation.ipynb) |

**References externes** :
- [SHAP Library](https://github.com/slundberg/shap) - Explications via valeurs de Shapley
- [InterpretML](https://github.com/interpretml/interpret) - Framework d'explicabilite Microsoft
- [Kaggle - Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) - Dataset
- [Fairlearn](https://fairlearn.org/) - Audit de biais et fairness

---

### C.7 - Detection de Fraude en Temps Reel

**Difficulte** : 3/5 | **Domaine** : ML

**Description** :
La detection de fraude est un probleme de classification fortement desequilibre (moins de 0.1% des transactions) avec des contraintes de latence. Le projet explore et compare des approches complementaires : Isolation Forest, Autoencoders, et GNN sur graphes de transactions. La gestion du desequilibre (SMOTE, class weights, focal loss) et l'evaluation avec des metriques adaptees (AUPRC, cout financier) sont centrales.

**Objectifs gradues** :
- **Minimum** : Isolation Forest + Autoencoder sur le dataset Kaggle Credit Card Fraud, metriques AUPRC
- **Bon** : Comparaison de 3+ methodes, gestion du desequilibre, analyse des faux positifs par cout financier
- **Excellent** : GNN sur graphe de transactions, detection en streaming (simulation temps reel), pipeline de decision avec seuils adaptatifs

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-24 | Autoencoders et detection d'anomalies | [QC-Py-24](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-24-Autoencoders-Anomaly.ipynb) |
| ML-4 | Evaluation de modeles (metriques) | [ML-4](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-4-Evaluation.ipynb) |

**References externes** :
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) - Dataset de reference (point de depart recommande)
- [PyOD (Outlier Detection)](https://github.com/yzhao062/pyod) - 40+ algorithmes de detection d'anomalies
- [PyOD Fraud Detection Example](https://pyod.readthedocs.io/en/latest/example.html) - Tutoriel
- [PyTorch Geometric](https://pyg.org/) - GNN pour graphes de transactions (niveau Excellent)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/) - Gestion du desequilibre de classes

[Retour a l'index des sujets](#index-des-sujets)

---

## Categorie D : Confidentialite et Securite du ML

Comment entrainer des modeles sans voir les donnees ? Sujet critique pour la banque et l'assurance (RGPD). Ces sujets explorent les techniques de Privacy-Preserving Machine Learning.

---

### D.1 - Federated Learning pour Prediction de Defaut Collaborative

**Difficulte** : 3/5 | **Domaine** : ML, Privacy

**Description** :
Systeme de Federated Learning permettant a plusieurs banques de collaborer pour entrainer un modele de prediction de defaut sans partager leurs donnees clients. Le modele voyage vers les donnees, apprend localement, et renvoie uniquement les mises a jour de poids (gradients) au serveur central. Le systeme simule plusieurs clients avec des donnees heterogenes (non-IID) et un serveur d'agregation.

**Objectifs gradues** :
- **Minimum** : Federated Averaging simple, 2-3 clients avec donnees synthetiques, modele lineaire
- **Bon** : Plus de clients, donnees heterogenes (non-IID), differential privacy pour l'agregation
- **Excellent** : Communication-efficient (compression de gradients), personalisation locale, attaque par inversion (evaluation privacy), comparaison avec centralise

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| ML-3 | Entrainement de modeles ML | [ML-3](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-3-Entrainement%26AutoML.ipynb) |

> **Note** : Ce sujet est principalement base sur des frameworks Python specialises (Flower, PySyft). Le notebook ML-3 fournit les bases d'entrainement ; les tutoriels Flower sont le vrai point de depart.

**References externes** :
- [Flower](https://flower.ai/) - Framework FL recommande (point de depart)
- [Flower Quickstart PyTorch](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html) - Tutoriel pas a pas
- [TensorFlow Federated](https://www.tensorflow.org/federated) - Framework FL Google
- [PySyft (OpenMined)](https://github.com/OpenMined/PySyft) - Privacy-preserving ML
- [Federated Learning Survey (Kairouz et al.)](https://arxiv.org/abs/1912.04977) - Reference theorique

---

### D.2 - Chiffrement Homomorphe (FHE) pour Agregation de Donnees Financieres

**Difficulte** : 4/5 | **Domaine** : Cryptographie, ML

**Description** :
Le Saint Graal de la privacy : effectuer des calculs (inference ML) directement sur des donnees chiffrees, sans jamais les dechiffrer. Le projet utilise le chiffrement homomorphe pour permettre l'agregation de donnees financieres sensibles (positions de risque, scoring) sans exposer les donnees brutes. Application concrete : scoring de credit sur donnees bancaires chiffrees.

**Objectifs gradues** :
- **Minimum** : Somme de nombres chiffres avec une librairie FHE (Microsoft SEAL, TenSEAL), demonstration de concept
- **Bon** : Operations plus complexes (moyenne, variance) sur donnees chiffrees, benchmark performance vs non-chiffre
- **Excellent** : Agregation de vecteurs de features, ML sur donnees chiffrees (classification simple avec Concrete-ML), analyse trade-off precision/performance

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
> **Note** : Pas de notebook de reference directe dans le cours. Ce sujet est entierement base sur des bibliotheques externes specialisees (Concrete-ML, SEAL). L'accent est sur l'apprentissage autonome d'une technologie emergente.

**References externes** :
- [Concrete ML (Zama)](https://github.com/zama-ai/concrete-ml) - Convertir modeles scikit-learn en equivalents chiffres (point de depart recommande)
- [Concrete ML Tutorials](https://docs.zama.ai/concrete-ml/tutorials) - Tutoriels officiels pas a pas
- [Microsoft SEAL](https://github.com/Microsoft/SEAL) - Bibliotheque FHE de reference
- [TenSEAL](https://github.com/OpenMined/TenSEAL) - Tensors chiffres pour PyTorch
- [Zama Blog - FHE for ML](https://www.zama.ai/blog) - Articles pedagogiques

---

### D.3 - Detection de Data Poisoning Adversarial

**Difficulte** : 3/5 | **Domaine** : ML, Securite

**Description** :
Systeme detectant les tentatives de data poisoning dans des modeles ML financiers. Un attaquant malveillant injecte des donnees corrompues pour biaiser le modele (par exemple, faire accepter des prets frauduleux ou fausser un modele de trading). Le systeme identifie ces donnees empoisonnees en analysant leur influence sur les parametres du modele et les predictions.

**Objectifs gradues** :
- **Minimum** : Simulation de poison attack simple (label flipping), detection basee sur statistiques de base
- **Bon** : Plusieurs types d'attaques (clean-label, backdoor), detection avec influence functions, evaluation de robustesse
- **Excellent** : Detection en ligne (streaming), defense proactive (data sanitization), evaluation sur cas d'usage financier (fraude, credit)

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| ML-3 | Entrainement de modeles (pipeline) | [ML-3](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-3-Entrainement%26AutoML.ipynb) |

> **Note** : Ce sujet utilise principalement le framework ART (IBM) pour les attaques et defenses. Le notebook ML-3 fournit les bases d'entrainement de modeles.

**References externes** :
- [ART (Adversarial Robustness Toolbox)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - Framework IBM (point de depart recommande)
- [ART Poisoning Tutorial](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_svm.ipynb) - Tutoriel data poisoning
- [SecML](https://secml.readthedocs.io/) - Security evaluation pour ML
- [Backdoor Attacks Survey (Li et al., 2022)](https://arxiv.org/abs/2007.08745) - Vue d'ensemble des attaques

[Retour a l'index des sujets](#index-des-sujets)

---

## Categorie E : Deep Learning et Architectures Modernes

Ces sujets explorent les architectures deep learning les plus recentes (2024-2026) appliquees a la finance : foundation models, transformers specialises, diffusion models, SSM (Mamba), GNN et PINNs. Ils s'appuient sur des publications recentes de NeurIPS, ICML et AAAI.

---

### E.1 - Foundation Models pour Series Financieres (Kronos)

**Difficulte** : 3/5 | **Domaine** : ML, Probas

**Description** :
Utiliser des foundation models pre-entraines specifiquement pour les series temporelles financieres afin de realiser du zero-shot ou few-shot forecasting de prix, volatilite ou indicateurs macro. Comparer les modeles generalistes (Chronos-2, TimesFM, Moirai-2.0) avec les modeles specialises finance (Kronos, Delphyne). Kronos (AAAI 2026) est le premier foundation model pre-entraine sur 12 milliards de K-lines de 45 bourses mondiales, avec 93% d'amelioration en RankIC sur les meilleurs concurrents.

C'est le "moment BERT" des series financieres (theme du workshop NeurIPS 2025).

**Objectifs gradues** :
- **Minimum** : Benchmark Kronos vs Chronos-2 en zero-shot sur le CAC40, metriques de prediction (MAE, RMSE)
- **Bon** : Comparaison zero-shot vs fine-tuned, evaluation sur plusieurs horizons temporels, analyse d'incertitude
- **Excellent** : Strategie de trading basee sur les signaux du modele, backtesting sur QuantConnect, generation de donnees synthetiques

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| ML-5 | Series temporelles ML.NET | [ML-5](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-5-TimeSeries.ipynb) |
| QC-Py-18 | Feature engineering ML | [QC-Py-18](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-18-ML-Features-Engineering.ipynb) |

**References externes** :
- [Kronos (AAAI 2026)](https://arxiv.org/abs/2508.02739) | [GitHub](https://github.com/shiyu-coder/Kronos) - Foundation model financier
- [Delphyne (Bloomberg, NeurIPS 2025)](https://arxiv.org/abs/2506.06288) - Modele Bloomberg
- [Chronos 2 (Amazon)](https://huggingface.co/amazon/chronos-2) - Modele generaliste
- [Moirai 2.0 (Salesforce)](https://arxiv.org/html/2511.11698v1) - Multi-variate forecasting
- [Blog Kronos (Kinlay 2026)](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/) - Vue d'ensemble

---

### E.2 - Transformers pour Limit Order Book (TLOB)

**Difficulte** : 4/5 | **Domaine** : ML

**Description** :
Appliquer des architectures transformer specialisees (TLOB, LiT) avec double mecanisme d'attention pour predire les mouvements de prix a partir de donnees high-frequency de carnets d'ordres (LOB). Le TLOB (2025) capture simultanement les dependances spatiales (entre niveaux de prix) et temporelles, depassant les approches CNN (DeepLOB). Le dataset FI-2010 est le benchmark standard ; les donnees LOBSTER sont disponibles gratuitement pour usage academique.

**Objectifs gradues** :
- **Minimum** : Reproduire TLOB sur le dataset FI-2010 (librement disponible), metriques de classification
- **Bon** : Comparaison TLOB vs DeepLOB (baseline CNN), analyse des poids d'attention
- **Excellent** : Test sur donnees crypto LOB (Binance API), visualisation des niveaux de prix et timesteps les plus predictifs

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-23 | Attention et Transformers | [QC-Py-23](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-23-Attention-Transformers.ipynb) |
| QC-Py-22 | Deep Learning LSTM | [QC-Py-22](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-22-Deep-Learning-LSTM.ipynb) |

**References externes** :
- [TLOB (arXiv 2025)](https://arxiv.org/abs/2502.15757) | [GitHub](https://github.com/LeonardoBerti00/TLOB) - Architecture de reference
- [LiT - Limit Order Book Transformer (2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full) - Architecture alternative
- [FI-2010 Dataset](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) - Benchmark LOB

---

### E.3 - Diffusion Models pour Donnees Financieres Synthetiques

**Difficulte** : 4/5 | **Domaine** : ML, Probas

**Description** :
Utiliser des modeles de diffusion (DDPM) pour generer des series temporelles financieres synthetiques realistes preservant les faits stylises : queues epaisses, clustering de volatilite, autocorrelation des rendements au carre. CoFinDiff (mars 2025) est un modele conditionnel controlable utilisant le cross-attention. Ce sujet resout un probleme pratique majeur : le manque de donnees pour le backtesting et le developpement de modeles.

**Objectifs gradues** :
- **Minimum** : DDPM simple pour generer des rendements quotidiens synthetiques, evaluation visuelle
- **Bon** : Evaluation quantitative (kurtosis, GARCH fit, autocorrelation), comparaison avec GAN et Monte Carlo
- **Excellent** : Modele conditionnel (CoFinDiff), augmentation de donnees pour backtesting d'une strategie, generation multi-actifs

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-18 | Feature engineering pour donnees financieres | [QC-Py-18](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-18-ML-Features-Engineering.ipynb) |
| QC-Py-24 | Autoencoders (modeles generatifs de base) | [QC-Py-24](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-24-Autoencoders-Anomaly.ipynb) |

> **Note** : Ce sujet avance est principalement base sur des tutoriels PyTorch externes. Les notebooks fournissent le contexte financier (features, autoencoders).

**References externes** :
- [Denoising Diffusion Tutorial (HuggingFace)](https://huggingface.co/blog/annotated-diffusion) - Tutoriel DDPM annote (point de depart recommande)
- [CoFinDiff (arXiv 2025)](https://arxiv.org/abs/2503.04164) - Diffusion controlable pour la finance
- [Synthetic Data for Finance (arXiv 2025)](https://arxiv.org/abs/2512.21791) - Survey
- [CFA Institute - Synthetic Data in Investment Management (2025)](https://rpc.cfainstitute.org/research/reports/2025/synthetic-data-in-investment-management) - Rapport industrie
- [FinDiff (tabular)](https://dl.acm.org/doi/fullHtml/10.1145/3604237.3626876) - Diffusion pour donnees tabulaires

---

### E.4 - Mamba/SSM pour Prediction Financiere

**Difficulte** : 3/5 | **Domaine** : ML

**Description** :
Appliquer Mamba -- un modele a espace d'etats selectif (SSM) avec complexite lineaire -- a la prediction de series financieres. Mamba filtre selectivement le bruit tout en capturant les dependances longue-portee, avec un avantage computationnel de 20-40x sur l'attention optimisee pour les longues sequences. CryptoMamba (2025) est la premiere architecture Mamba specifiquement concue pour la crypto. C'est le paradigme "post-transformer" pour les donnees sequentielles.

**Objectifs gradues** :
- **Minimum** : Adapter Mamba pour prediction de rendements Bitcoin, comparaison vs LSTM
- **Bon** : Comparaison Mamba vs Transformer vs ARIMA, evaluation sur plusieurs cryptos pour tester la generalisation
- **Excellent** : Analyse du mecanisme selectif (quelles parties de la sequence sont retenues/oubliees), strategie de trading, backtesting

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-22 | Deep Learning LSTM (baseline) | [QC-Py-22](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-22-Deep-Learning-LSTM.ipynb) |
| QC-Py-23 | Attention et Transformers | [QC-Py-23](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-23-Attention-Transformers.ipynb) |

**References externes** :
- [CryptoMamba (2025)](https://ui.adsabs.harvard.edu/abs/2025arXiv250101010S/abstract) - Mamba pour crypto
- [Mamba Architecture (GitHub)](https://github.com/state-spaces/mamba) - Implementation officielle
- [MambaTS (ICLR 2025)](https://openreview.net/pdf?id=vEtDApqkNR) - SSM pour series temporelles
- [Survey SSM (S4 to Mamba)](https://arxiv.org/abs/2503.18970) - Vue d'ensemble

---

### E.5 - GNN pour Construction de Portefeuille

**Difficulte** : 4/5 | **Domaine** : ML, Probas

**Description** :
Modeliser les relations inter-actifs (correlations, chaines d'approvisionnement, secteurs) comme des graphes dynamiques et utiliser des GNN pour apprendre des poids de portefeuille qui tiennent compte de ces structures relationnelles. Des travaux recents (Nature 2025, SSRN 2025) montrent 16.8% de rendement annualise et 1.34 de Sharpe sur S&P500/NASDAQ avec des architectures graph attention + DRL heterogene.

**Objectifs gradues** :
- **Minimum** : Graphe de correlation entre 30 actions (DJIA), GCN simple pour prediction de rendements
- **Bon** : GAT (Graph Attention Network), portefeuille Markowitz avec rendements predits par GNN, comparaison vs equal-weight
- **Excellent** : Graphe dynamique (evolution temporelle des correlations), GNN + DRL pour allocation, backtesting multi-periodes

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-21 | Optimisation de portefeuille ML | [QC-Py-21](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-21-Portfolio-Optimization-ML.ipynb) |
| QC-Py-19 | ML supervise pour classification | [QC-Py-19](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-19-ML-Supervised-Classification.ipynb) |

**References externes** :
- [Graph Attention Portfolio (Nature 2025)](https://www.nature.com/articles/s41598-025-32408-w) - Reference recente
- [GNN Deep Portfolio (Springer)](https://link.springer.com/article/10.1007/s00521-023-08862-w) - GNN pour portefeuille
- [Survey AI in Quant Investment (arXiv 2025)](https://arxiv.org/html/2503.21422v1) - Vue d'ensemble
- [PyTorch Geometric](https://pyg.org/) - Bibliotheque GNN

---

### E.6 - PINNs pour Pricing d'Options

**Difficulte** : 4/5 | **Domaine** : Probas, ML

**Description** :
Resoudre l'equation aux derivees partielles (EDP) de Black-Scholes (et extensions : Heston, Merton jump-diffusion) avec des Physics-Informed Neural Networks (PINNs) qui integrent l'EDP financiere comme contrainte de loss. Les PINNs offrent une solution mesh-free pour le pricing de derives complexes, avec 12.5% d'amelioration sur calls NASDAQ et 59% sur puts americaines vs methodes traditionnelles. Le sujet est pedagogiquement tres riche : il fait le pont entre physique/math et deep learning.

**Objectifs gradues** :
- **Minimum** : PINN en PyTorch pour resoudre l'EDP Black-Scholes (call europeen), comparaison avec solution analytique
- **Bon** : Extension aux puts americaines (probleme a frontiere libre), visualisation de la surface prix(spot, maturite)
- **Excellent** : Extension au modele de Heston ou Merton, pricing d'options exotiques, analyse de convergence

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Black-Scholes et probabilites (reference theorique) | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

> **Note** : Aucun notebook du cours ne couvre directement les PINNs. Le notebook Infer-101 fournit le contexte theorique (Black-Scholes). Pour l'implementation, suivre les tutoriels DeepXDE et le cours MathWorks ci-dessous.

**References externes** :
- [DeepXDE](https://deepxde.readthedocs.io/) - Bibliotheque PINNs Python (tutoriels inclus : diffusion, Burgers, Black-Scholes)
- [DeepXDE Examples](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward.html) - Galerie d'exemples PDE forward/inverse
- [PINN Option Pricing (arXiv)](https://arxiv.org/abs/2312.06711) - Paper de reference
- [MathWorks PINN Tutorial (2025)](https://blogs.mathworks.com/finance/2025/01/07/physics-informed-neural-networks-pinns-for-option-pricing/) - Tutoriel detaille avec code
- [PINNs GitHub (MATLAB)](https://github.com/matlab-deep-learning/PINNsOptionPricing) - Implementation MATLAB
- [PyTorch PINN Tutorial](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/) - Introduction pedagogique

[Retour a l'index des sujets](#index-des-sujets)

---

## Categorie F : IA Agents et LLMs pour la Finance

Ces sujets explorent l'utilisation des grands modeles de langage (LLM) et des systemes multi-agents pour des taches financieres : trading, sentiment, generation de scenarios, fine-tuning et raisonnement hybride.

---

### F.1 - Multi-Agent LLM Trading (TradingAgents)

**Difficulte** : 4/5 | **Domaine** : GameTheory, ML

**Description** :
Construire un systeme multi-agents ou des LLMs specialises (analyste fondamental, analyste sentiment, analyste technique, risk manager, trader) collaborent pour prendre des decisions de trading, imitant la structure d'une salle de marche reelle. TradingAgents (NeurIPS 2025 workshop) propose 7 roles specialises avec un mecanisme de debat bull/bear. Le V0.2.0 supporte plusieurs providers LLM.

**Objectifs gradues** :
- **Minimum** : 3 agents simplifies (analyste + risk manager + trader) avec LLM local (Qwen3) ou API, backtesting simple
- **Bon** : Mecanisme de debat, comparaison single-agent vs multi-agent, analyse qualitative des decisions
- **Excellent** : 5+ agents, integration de donnees multi-sources (news, technique, fondamental), backtesting QuantConnect, analyse de la valeur ajoutee du debat

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-17 | Multi-Agent Reinforcement Learning | [GT-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL.ipynb) |
| Lab13 | Recherche web et agents SOTA | [Lab13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/AgenticDataScience/Day6-MLE-Star/Lab13-Web-Search-SOTA.ipynb) |

**References externes** :
- [TradingAgents (NeurIPS 2025)](https://arxiv.org/abs/2412.20138) | [GitHub](https://github.com/TauricResearch/TradingAgents) - Framework de reference
- [FINCON (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/f7ae4fe91d96f50abc2211f09b6a7e49-Paper-Conference.pdf) - Manager-analyst pipeline
- [LLM Agent Trading Survey](https://arxiv.org/html/2408.06361v1) - Vue d'ensemble

---

### F.2 - LLM Sentiment Alpha Generation (DK-CoT)

**Difficulte** : 3/5 | **Domaine** : ML, NLP

**Description** :
Utiliser des LLMs (FinBERT, Llama, GPT-4) avec des techniques de prompting avancees (Domain Knowledge Chain-of-Thought) pour extraire du sentiment de news financieres, earnings calls et reseaux sociaux, puis convertir ces signaux en alpha tradeable. Des etudes recentes documentent 312 points de base d'alpha annuel via des signaux linguistiques, montant a 476bp pendant les saisons de resultats.

**Objectifs gradues** :
- **Minimum** : FinBERT pour scorer le sentiment de titres de news financieres, portefeuille long-short basique
- **Bon** : Comparaison FinBERT vs GPT-4 zero-shot vs DK-CoT, backtesting sur 1 an, analyse des periodes de surperformance
- **Excellent** : Multi-source (news + Twitter + earnings), strategie event-driven, integration QuantConnect, cout d'execution

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Lab13 | Recherche web et SOTA | [Lab13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/AgenticDataScience/Day6-MLE-Star/Lab13-Web-Search-SOTA.ipynb) |
| QC-Py-17 | Analyse de sentiment | [QC-Py-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-17-Sentiment-Analysis.ipynb) |
| QC-Py-26 | LLM Trading Signals | [QC-Py-26](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-26-LLM-Trading-Signals.ipynb) |

**References externes** :
- [DK-CoT Sentiment (Springer 2025)](https://link.springer.com/article/10.1007/s10791-025-09573-7) - Domain Knowledge CoT
- [LLMs in Equity Markets (Frontiers AI 2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) - Survey
- [FinBen Benchmark](https://arxiv.org/abs/2402.12659) | [Leaderboard HuggingFace](https://huggingface.co/blog/leaderboard-finbench) - Evaluation

---

### F.3 - FinGPT Fine-Tuning pour Taches Financieres

**Difficulte** : 3/5 | **Domaine** : ML, NLP

**Description** :
Fine-tuner un LLM open-source (Llama, Mistral) sur des donnees financieres avec LoRA/QLoRA pour creer un modele financier specialise, puis evaluer sur le benchmark FinBen (42 datasets, 24 taches, 8 domaines). FinGPT (AI4Finance) est l'alternative open-source a BloombergGPT ($2M+), avec un cout de fine-tuning d'environ $300. La democratisation de l'IA financiere est en marche.

**Objectifs gradues** :
- **Minimum** : Fine-tune QLoRA d'un modele 7-8B sur Financial PhraseBank (sentiment), evaluation basique
- **Bon** : Evaluation sur plusieurs taches FinBen (sentiment, NER, QA), comparaison fine-tuned vs base vs GPT-4 zero-shot
- **Excellent** : Analyse cout/performance, publication sur HuggingFace, participation au FinAI Contest 2025

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Lab13 | Recherche web et SOTA | [Lab13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/AgenticDataScience/Day6-MLE-Star/Lab13-Web-Search-SOTA.ipynb) |
| QC-Py-26 | LLM Trading Signals | [QC-Py-26](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-26-LLM-Trading-Signals.ipynb) |

**References externes** :
- [FinGPT (AI4Finance)](https://github.com/AI4Finance-Foundation/FinGPT) | [Paper](https://arxiv.org/abs/2306.06031) - Framework open-source
- [FinBen Benchmark](https://arxiv.org/abs/2402.12659) | [thefin.ai](https://www.thefin.ai/dataset-benchmark/finben) - Evaluation standardisee
- [Open FinLLM Leaderboard](https://huggingface.co/blog/leaderboard-finbench) - Classement continu
- [FinAI Contest 2025](https://open-finance-lab.github.io/FinAI_Contest_2025/) - Competition

---

### F.4 - LLMs pour Generation de Scenarios Macroeconomiques

**Difficulte** : 3/5 | **Domaine** : ML

**Description** :
Utiliser des LLMs pour generer des scenarios macroeconomiques coherents (inflation, croissance, taux, chomage) afin de tester la robustesse de portefeuilles. Les scenarios generes doivent etre realistes (respecter les correlations macro-connues), diversifies (couvrir les extremes) et actionables (utilisables par un moteur de risque). Ce sujet combine LLMs et risk management, un duo tres demande en entreprise.

**Objectifs gradues** :
- **Minimum** : Generation de scenarios simples avec un LLM (prompting), validation de coherence manuelle
- **Bon** : Pipeline de generation structuree (JSON), contraintes de coherence (correlations), stress scenarios
- **Excellent** : Evaluation de la couverture des risques, integration avec moteur de risque, interface pour risk managers, fine-tuning optionnel

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Lab13 | Recherche web et SOTA | [Lab13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/AgenticDataScience/Day6-MLE-Star/Lab13-Web-Search-SOTA.ipynb) |
| Lab7 | Agent d'analyse de donnees | [Lab7](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/PythonAgentsForDataScience/Day3/Labs/Lab7-Data-Analysis-Agent/Lab7-Data-Analysis-Agent.ipynb) |

**References externes** :
- [LangChain](https://www.langchain.com/) - Framework pour pipelines LLM
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/) - Tutoriels complets (chains, agents, RAG)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) - Generation JSON structuree
- [AutoGen (Microsoft)](https://github.com/microsoft/autogen) - Multi-agents conversationnels pour generation structuree
- [CrewAI](https://github.com/crewAIInc/crewAI) - Framework multi-agents avec roles specialises
- [IMF Scenario Generation](https://www.imf.org/en/Publications/WP/Issues/2024/04/12/Scenario-Analysis-with-AI) - Approche IMF pour les scenarios macro

---

### F.5 - Neurosymbolic AI pour Decisions de Credit

**Difficulte** : 4/5 | **Domaine** : ML, Symbolic AI

**Description** :
Systeme hybride neurosymbolique combinant un reseau de neurones (pour la prediction) et une couche symbolique (pour l'explication generative). Contrairement au XAI post-hoc classique (SHAP, LIME), l'approche neurosymbolique integre le raisonnement explicable directement dans l'architecture du modele. Le systeme explique ses decisions de credit en langage naturel avec des regles formelles verifiables.

**Objectifs gradues** :
- **Minimum** : Neural network + regles symboliques simples post-hoc, explications basiques en texte
- **Bon** : Integration neurone-symbole pendant l'entrainement, explications contrefactuelles
- **Excellent** : Architecture neurosymbolique complete (DeepProbLog ou similaire), explications generatives avec LLM, evaluation humaine

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Lab3 | CV Screening (classification) | [Lab3](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/PythonAgentsForDataScience/Day2/Labs/Lab3-CV-Screening/Lab3-CV-Screening.ipynb) |
| Infer-19 | Systemes experts et decision | [Infer-19](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-19-Decision-Expert-Systems.ipynb) |

**References externes** :
- [DeepProbLog](https://github.com/ML-KULeuven/deepproblog) - Programmation logique probabiliste + deep learning
- [NeurASP](https://github.com/azreasoners/NeurASP) - Neuro-symbolic Answer Set Programming
- [SHAP](https://github.com/slundberg/shap) - Pour comparaison XAI classique

[Retour a l'index des sujets](#index-des-sujets)

---

## Categorie G : Recherche et Innovation

Sujets exploratoires bases sur des publications recentes (NeurIPS 2024-2025, ICML 2025, AAAI 2026). Pour les etudiants qui veulent toucher a la frontiere de la recherche. Difficulte plus elevee, mais potentiel d'originalite maximal.

---

### G.1 - Causal ML pour Asset Pricing (EconML/DoWhy)

**Difficulte** : 3/5 | **Domaine** : Probas, ML

**Description** :
Appliquer les methodes d'inference causale (Double Machine Learning, forets causales) pour identifier les vrais drivers causaux des rendements d'actifs, au-dela de la simple correlation. Le passage de "predire" a "comprendre pourquoi" est un changement fondamental en finance quantitative. EconML et DoWhy sont des bibliotheques matures pour le DML et les forets causales.

**Objectifs gradues** :
- **Minimum** : Double ML avec EconML pour estimer l'effet causal des surprises de resultats sur les rendements, comparaison vs regression naive
- **Bon** : Forets causales pour effets heterogenes par secteur, analyse de sensibilite aux confounders
- **Excellent** : Pipeline end-to-end causal (discovery + estimation), intervention analysis (what-if), visualisation des graphes causaux

**Notebooks de reference** :

> **Note** : Ce sujet s'appuie principalement sur les bibliotheques EconML et DoWhy. Aucun notebook du cours ne couvre directement l'inference causale. Les tutoriels officiels ci-dessous constituent les meilleures ressources de demarrage.

**References externes** :
- [EconML (Microsoft)](https://github.com/py-why/EconML) - Double ML et forets causales
- [EconML User Guide](https://econml.azurewebsites.net/spec/estimation.html) - Tutoriel complet avec exemples
- [DoWhy (Microsoft)](https://github.com/py-why/dowhy) - Framework causal complet
- [DoWhy Getting Started](https://www.pywhy.org/dowhy/main/getting_started/index.html) - Tutoriel d'introduction avec notebooks
- [Causal Inference for The Brave and True](https://matheusfacure.github.io/python-causality-handbook/) - Livre en ligne gratuit (Python)
- [KDD 2025 Causal ML Pipeline](https://causal-machine-learning.github.io/kdd2025-workshop/papers/12.pdf) - Pipeline end-to-end
- [DML for Stock Mispricing (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0275531924004616) - Application finance

---

### G.2 - GNN Risque Systemique et Contagion Financiere

**Difficulte** : 4/5 | **Domaine** : GameTheory, ML

**Description** :
Modeliser le systeme financier comme un graphe dynamique (banques = noeuds, expositions = aretes) et utiliser des GNN pour predire le risque systemique et les voies de contagion. Les travaux recents (NeurIPS 2025, GNN-MAM 2025) montrent que les GNN surpassent les mesures de centralite classiques pour la prediction de defauts en cascade. Directement pertinent pour les frameworks reglementaires post-2008 (Bale III/IV).

**Objectifs gradues** :
- **Minimum** : Graphe de reseau bancaire (simule ou donnees publiques), GCN pour predire les defauts
- **Bon** : Simulation de contagion en cascade, comparaison GNN vs centralite classique (degre, betweenness)
- **Excellent** : Graphe multi-couches (interbank + marchE + derives), analyse contrefactuelle, stress testing

**Notebooks de reference** :

> **Note** : Aucun notebook du cours ne couvre directement les GNN. Ce sujet s'appuie sur PyTorch Geometric et les tutoriels ci-dessous pour la partie technique.

**References externes** :
- [PyTorch Geometric](https://pyg.org/) - Bibliotheque GNN de reference
- [PyG Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) - Introduction complete avec notebooks
- [PyG Colab Notebooks](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html) - Tutoriels interactifs (GCN, GAT, GraphSAGE)
- [Systemic Risk Radar (arXiv 2025)](https://arxiv.org/html/2512.17185) - GNN + Knowledge Graph
- [GNN-MAM (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S1110016825007641) - Multi-Attention GNN
- [Deep Graph Learning for Systemic Risk](https://www.sciencedirect.com/science/article/pii/S2667305323000650) - Reference
- [Stanford CS224W](http://web.stanford.edu/class/cs224w/) - Cours Machine Learning with Graphs (slides + videos)

---

### G.3 - RL Market Making et Execution Optimale

**Difficulte** : 4/5 | **Domaine** : GameTheory, ML

**Description** :
Entrainer des agents RL pour agir comme market makers (posting bid/ask) ou executer de gros ordres de facon optimale en minimisant l'impact de marche. POW-dTS (2025) propose le Policy Weighting avec Thompson Sampling pour l'adaptation aux changements de regime. Le framework FinRL fournit des environnements Gymnasium prets a l'emploi.

**Objectifs gradues** :
- **Minimum** : Agent DQN/PPO pour execution d'un gros ordre (10 000 actions), comparaison vs TWAP
- **Bon** : Environnement LOB simule, analyse de l'apprentissage (splitting, timing), metriques d'implementation shortfall
- **Excellent** : Market making (bid/ask quotes), multi-objectif (profit vs risque vs spread), test sur donnees reelles

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-17 | Multi-Agent RL | [GT-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL.ipynb) |
| QC-Py-25 | Reinforcement Learning | [QC-Py-25](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-25-Reinforcement-Learning.ipynb) |

**References externes** :
- [FinRL (AI4Finance)](https://github.com/AI4Finance-Foundation/FinRL) - Environnements RL finance
- [RL Market Making (arXiv 2025)](https://arxiv.org/abs/2507.18680) - Paper recent
- [RL Optimal Execution (arXiv 2024)](https://arxiv.org/abs/2411.06389) - Execution optimale
- [Market Making RL (GitHub)](https://github.com/KodAgge/Reinforcement-Learning-for-Market-Making) - Implementation

---

### G.4 - World Models pour Trading (DreamerV3)

**Difficulte** : 4/5 | **Domaine** : GameTheory, ML

**Description** :
Systeme de trading utilisant un "world model" qui apprend une dynamique de marche simulee, puis planifie dans ce monde simule. Inspire de DreamerV3 (architecture SOTA en RL model-based), le systeme apprend un modele latent du marche et utilise l'imagination pour evaluer des strategies sans executer de trades reels. Cela reduit considerablement le sample complexity du RL.

**Objectifs gradues** :
- **Minimum** : World model simple (MLP latent dynamics), planification MPC basique, environnement de marche simule
- **Bon** : Architecture complete Dreamer (world model + actor-critic), apprentissage de la dynamique, evaluation
- **Excellent** : Transfer de la simulation aux donnees reelles, analyse de la qualite du monde appris, comparaison avec RL direct (model-free)

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-17 | Multi-Agent RL | [GT-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL.ipynb) |
| QC-Py-25 | Reinforcement Learning | [QC-Py-25](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-25-Reinforcement-Learning.ipynb) |

**References externes** :
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) - Architecture de reference
- [DreamerV3 GitHub](https://github.com/danijar/dreamerv3) - Implementation officielle JAX
- [DreamerV3-torch](https://github.com/NM512/dreamerv3-torch) - Portage PyTorch (plus accessible)
- [MBRL-Lib (Meta)](https://github.com/facebookresearch/mbrl-lib) - Bibliotheque Model-Based RL (benchmarks, tutoriels)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Algorithmes RL model-free (baseline de comparaison)
- [Gymnasium](https://gymnasium.farama.org/) - Environnements RL standard

---

### G.5 - Imitation-RL pour Controle Stochastique en Finance (FinFlowRL)

**Difficulte** : 5/5 | **Domaine** : Probas, GameTheory, ML

**Description** :
Combiner l'imitation learning (apprendre de demonstrations expertes) avec le reinforcement learning pour resoudre des problemes de controle stochastique en finance : hedging optimal, rebalancement de portefeuille, gestion d'inventaire. FinFlowRL (NeurIPS 2025 GenAI Finance workshop) propose un framework novateur pour le controle adaptatif. L'initialisation par imitation offre un entrainement plus stable que le RL pur. C'est le sujet le plus ambitieux de ce catalogue.

**Objectifs gradues** :
- **Minimum** : Imitation learning pour initialiser un agent RL a partir d'une heuristique simple (rebalancement a proportion constante)
- **Bon** : Fine-tuning RL pour ameliorer le Sharpe ratio, comparaison RL pur vs IL-initialise (vitesse de convergence)
- **Excellent** : Application a un probleme 3+ actifs, integration PINNs pour HJB, analyse theorique de la convergence

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-17 | Multi-Agent RL | [GT-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL.ipynb) |
| Infer-101 | Probabilites et controle stochastique | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |

**References externes** :
- [FinFlowRL (NeurIPS 2025 GenAI Finance Workshop)](https://sites.google.com/view/neurips-25-gen-ai-in-finance/accepted-papers) - Paper de reference
- [Heuristic-guided Inverse RL (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/1054.pdf) - Approche complementaire
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Algorithmes RL
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - Environnements finance

---

### G.6 - GFlowNets pour Generation de Portefeuilles

**Difficulte** : 4/5 | **Domaine** : Probas, ML

**Description** :
Les GFlowNets (Generative Flow Networks, Yoshua Bengio) sont une nouvelle famille de modeles generatifs concus pour echantillonner des objets composites (molecules, graphes, portefeuilles) proportionnellement a une recompense. Application a la generation de portefeuilles diversifies proportionnellement a leur rendement ajuste au risque. Contrairement au RL classique qui cherche l'optimum unique, les GFlowNets echantillonnent l'ensemble des bonnes solutions.

**Objectifs gradues** :
- **Minimum** : GFlowNet simple sur un environnement de grille, demonstration du concept d'echantillonnage proportionnel
- **Bon** : Application a la selection d'actifs (sous-ensembles de N actions), comparaison vs optimisation classique
- **Excellent** : Generation de portefeuilles diversifies avec contraintes, analyse de la diversite des solutions, backtesting

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Inference bayesienne et echantillonnage | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |
| QC-Py-21 | Optimisation de portefeuille | [QC-Py-21](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-21-Portfolio-Optimization-ML.ipynb) |

**References externes** :
- [Tutoriel GFlowNet (Mila)](https://mila.quebec/fr/article/gflownet-tutorial) - Introduction par le lab de Bengio
- [TorchGFN Library](https://github.com/GFNOrg/torchgfn) - Implementation PyTorch
- [GFlowNet Foundations (Bengio et al., 2023)](https://arxiv.org/abs/2111.09266) - Paper fondateur

[Retour a l'index des sujets](#index-des-sujets)

---

## Categorie H : Trading Algorithmique avec QuantConnect

Ces sujets utilisent la plateforme professionnelle [QuantConnect](https://www.quantconnect.com/) (moteur LEAN) pour concevoir, backtester et deployer des strategies de trading algorithmique. QuantConnect est utilise par des hedge funds et offre des donnees historiques de haute qualite, un IDE cloud et un moteur de backtesting robuste. Les 27 notebooks Python du cours couvrent de A a Z le workflow QuantConnect.

**Code promo** : `education2025` pour un acces gratuit aux fonctionnalites trading firm.

**Ressources partagees** : Les strategies existantes dans le cours incluent 40+ projets d'implementation et des bibliotheques Python partagees (backtest_helpers, features, indicators, ml_utils, plotting).

---

### H.1 - Strategie Alpha ML sur QuantConnect

**Difficulte** : 3/5 | **Domaine** : ML, QuantConnect

**Description** :
Concevoir et backtester une strategie de trading basee sur des modeles ML (classification, regression) sur la plateforme QuantConnect. Le projet utilise le framework Alpha du moteur LEAN pour generer des signaux d'achat/vente a partir de features financieres (indicateurs techniques, fondamentaux, sentiment). L'objectif est de battre un benchmark (S&P 500, buy-and-hold) sur une periode de 5+ ans.

**Objectifs gradues** :
- **Minimum** : Modele ML simple (Random Forest) avec features techniques, backtesting 3 ans, metriques de base (Sharpe, drawdown)
- **Bon** : Feature engineering avancee (multi-timeframe, fondamentaux), walk-forward validation, analyse de robustesse
- **Excellent** : Ensemble de modeles, selection dynamique de features, soumission Quant League, analyse des couts de transaction

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-19 | ML Supervise - Classification | [QC-Py-19](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-19-ML-Supervised-Classification.ipynb) |
| QC-Py-20 | ML Regression et Prediction | [QC-Py-20](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-20-ML-Regression-Prediction.ipynb) |
| QC-Py-21 | Optimisation de Portefeuille ML | [QC-Py-21](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-21-Portfolio-Optimization-ML.ipynb) |
| QC-Py-18 | Feature Engineering | [QC-Py-18](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-18-ML-Features-Engineering.ipynb) |

**References externes** :
- [QuantConnect Documentation](https://www.quantconnect.com/docs/) - Documentation officielle
- [LEAN Engine (GitHub)](https://github.com/QuantConnect/Lean) - Moteur open-source
- [QuantConnect Tutorials](https://www.quantconnect.com/tutorials/) - Tutoriels officiels

---

### H.2 - Deep RL Trading avec QuantConnect

**Difficulte** : 4/5 | **Domaine** : ML, GameTheory, QuantConnect

**Description** :
Entrainer un agent de Deep Reinforcement Learning (DQN, PPO, SAC) pour gerer un portefeuille multi-actifs sur QuantConnect. L'agent apprend a prendre des decisions buy/sell/hold en maximisant le Sharpe ratio cumule. Le defi principal est de gerer l'environnement non-stationnaire des marches financiers et d'eviter l'overfitting au backtesting.

**Objectifs gradues** :
- **Minimum** : Agent DQN simple sur 3-5 actifs, backtesting 3 ans, comparaison vs buy-and-hold
- **Bon** : PPO avec observation space enrichi (indicateurs techniques + volume), reward shaping pour Sharpe, walk-forward
- **Excellent** : SAC avec continuous actions (tailles de position), multi-asset, integration avec FinRL, Quant League submission

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-25 | Reinforcement Learning | [QC-Py-25](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-25-Reinforcement-Learning.ipynb) |
| QC-Py-22 | Deep Learning LSTM | [QC-Py-22](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-22-Deep-Learning-LSTM.ipynb) |
| GT-17 | Multi-Agent RL | [GT-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL.ipynb) |

**References externes** :
- [FinRL (AI4Finance)](https://github.com/AI4Finance-Foundation/FinRL) - Environnements RL finance
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Algorithmes RL
- [Gymnasium](https://gymnasium.farama.org/) - Environnements standard

---

### H.3 - Composite AlphaModel Framework

**Difficulte** : 3/5 | **Domaine** : ML, QuantConnect

**Description** :
Construire un framework de strategies composites utilisant l'architecture AlphaModel de QuantConnect. Le principe : combiner plusieurs signaux alpha (momentum, mean-reversion, sentiment, ML) dans un modele composite avec ponderation dynamique. Les meilleurs hedge funds fonctionnent ainsi : pas une seule strategie, mais un ensemble pondere adaptatif. Le cours contient deja des exemples de strategies composites (Fama-French + AllWeather).

**Objectifs gradues** :
- **Minimum** : 2 AlphaModels simples combines, backtesting, comparaison composite vs individuel
- **Bon** : 3+ AlphaModels avec ponderation basee sur performance recente, Portfolio Construction Model, risk management
- **Excellent** : Ponderation adaptative par regime de marche, selection dynamique des alphas, walk-forward validation

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-13 | Alpha Models | [QC-Py-13](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-13-Alpha-Models.ipynb) |
| QC-Py-14 | Portfolio Construction et Execution | [QC-Py-14](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-14-Portfolio-Construction-Execution.ipynb) |
| QC-Py-15 | Optimisation de Parametres | [QC-Py-15](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-15-Parameter-Optimization.ipynb) |

**References externes** :
- [QuantConnect Alpha Framework](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/alpha/key-concepts) - Documentation Alpha Models
- [LEAN Engine](https://github.com/QuantConnect/Lean) - Architecture framework

---

### H.4 - Regime Switching et Allocation Adaptative

**Difficulte** : 3/5 | **Domaine** : ML, Probas, QuantConnect

**Description** :
Construire une strategie qui detecte automatiquement les regimes de marche (bull, bear, haute volatilite, basse volatilite) et adapte son allocation en consequence. En regime risk-on, la strategie privilegiera les actifs risques ; en risk-off, elle se refugiera sur les obligations et l'or. La detection peut utiliser des HMM, des indicateurs de marche ou du ML.

**Objectifs gradues** :
- **Minimum** : Detection de regime simple (VIX threshold, moyenne mobile), allocation binaire (equities vs bonds)
- **Bon** : HMM pour 3+ regimes, allocation graduelle, backtesting multi-periodes incluant des crises
- **Excellent** : Detection ML (VAE, clustering), portefeuille multi-assets (equities, bonds, gold, commodities), walk-forward

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-10 | Risk et Portfolio Management | [QC-Py-10](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-10-Risk-Portfolio-Management.ipynb) |
| QC-Py-11 | Indicateurs Techniques | [QC-Py-11](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-11-Technical-Indicators.ipynb) |
| QC-Py-08 | Strategies Multi-Asset | [QC-Py-08](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-08-Multi-Asset-Strategies.ipynb) |

**References externes** :
- [hmmlearn](https://hmmlearn.readthedocs.io/) - HMM en Python
- [QuantConnect Risk Management](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/risk-management/key-concepts) - Documentation Risk Management

---

### H.5 - Options Strategies Automatisees (Wheel/Covered Call)

**Difficulte** : 3/5 | **Domaine** : ML, QuantConnect

**Description** :
Automatiser des strategies d'options classiques (Wheel Strategy, Covered Call, Iron Condor) sur QuantConnect. La Wheel Strategy consiste a vendre des puts cash-secured puis, si assignee, a vendre des calls couverts sur les actions recues. Ces strategies sont populaires chez les investisseurs income-oriented et se pretent bien a l'automatisation. Les notebooks QC couvrent le trading d'options et les projets existants incluent des implementations Wheel et Options-VGT.

**Objectifs gradues** :
- **Minimum** : Covered Call automatise sur un ETF (SPY), backtesting 3 ans, comparaison vs buy-and-hold
- **Bon** : Wheel Strategy complete (put selling + covered call), selection de strike basee sur les Grecs, gestion du risque
- **Excellent** : Multi-underlying, Iron Condor adaptatif, optimisation du strike/expiry par ML, analyse PnL detaillee

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-06 | Options Trading | [QC-Py-06](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-06-Options-Trading.ipynb) |
| QC-Py-09 | Order Types | [QC-Py-09](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-09-Order-Types.ipynb) |
| QC-Py-12 | Backtesting Analysis | [QC-Py-12](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-12-Backtesting-Analysis.ipynb) |

**References externes** :
- [QuantConnect Options Documentation](https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/equity-options) - Reference officielle
- [The Wheel Strategy (Investopedia)](https://www.investopedia.com/the-wheel-options-trading-strategy-7494445) - Explication detaillee

---

### H.6 - Walk-Forward Analysis et Robustesse de Strategies

**Difficulte** : 3/5 | **Domaine** : ML, QuantConnect

**Description** :
Construire un framework de validation robuste pour strategies de trading : walk-forward analysis, Monte Carlo permutation tests, analyse de sensibilite aux parametres. L'objectif est de distinguer les strategies genuinement performantes de celles qui ne font que de l'overfitting sur le backtesting. Ce sujet est meta : il s'applique a toutes les autres strategies QC et represente une competence fondamentale en finance quantitative.

**Objectifs gradues** :
- **Minimum** : Walk-forward analysis (train/test rolling) sur une strategie simple, rapport de metriques par fenetre
- **Bon** : Monte Carlo permutation test (strategie vs random), analyse de sensibilite parametrique (heatmap), multiple testing correction
- **Excellent** : Framework reutilisable pour valider n'importe quelle strategie, deflated Sharpe ratio, rapport automatise

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| QC-Py-27 | Production et Deploiement | [QC-Py-27](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-27-Production-Deployment.ipynb) |
| QC-Py-15 | Optimisation de Parametres | [QC-Py-15](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-15-Parameter-Optimization.ipynb) |
| QC-Py-12 | Backtesting Analysis | [QC-Py-12](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/QuantConnect/Python/QC-Py-12-Backtesting-Analysis.ipynb) |

**References externes** :
- [Advances in Financial ML (Lopez de Prado, 2018)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - Reference sur la robustesse
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) - Correction statistique
- [QuantConnect Optimization](https://www.quantconnect.com/docs/v2/cloud-platform/optimization) - Documentation

[Retour a l'index des sujets](#index-des-sujets)

---

## Categorie I : Sujets Non-Finance et Transversaux

Ces sujets ne sont pas specifiques a la finance mais couvrent des concepts fondamentaux en IA/ML applicables a de nombreux domaines. Ils permettent aux etudiants ayant des interets au-dela de la finance pure d'explorer des problemes de jeux, de sante, de sport ou de langage.

---

### I.1 - TrueSkill et Matchmaking Competitif

**Difficulte** : 3/5 | **Domaine** : Probas

**Description** :
Le classement de joueurs dans les jeux en ligne (Xbox Live, LoL, Chess) est un probleme probabiliste complexe. Au-dela du simple systeme ELO, le systeme TrueSkill utilise des graphes de facteurs pour modeliser l'incertitude sur la competence de chaque joueur (une gaussienne avec moyenne et variance). Le projet implemente un moteur d'inference (Expectation Propagation ou Variational Inference) pour mettre a jour les scores apres chaque match.

**Objectifs gradues** :
- **Minimum** : Implementation TrueSkill basique, mise a jour des scores pour matchs 1v1, visualisation de la convergence
- **Bon** : Gestion des equipes heterogenes, draw margin (probabilite de nul), comparaison avec ELO
- **Excellent** : Dynamique temporelle (progression/regression), TrueSkill 2 (contexte Xbox), application a des donnees reelles (Kaggle)

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Introduction Infer.NET et inference | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |
| Infer-10 | Crowdsourcing (modeles similaires) | [Infer-10](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-10-Crowdsourcing.ipynb) |

**References externes** :
- [TrueSkill (Microsoft Research)](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) - Projet original
- [TrueSkill 2 (Paper)](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/) - Version amelioree
- [MBML Book - TrueSkill Chapter](http://mbmlbook.com/TrueSkill.html) - Tutoriel detaille

---

### I.2 - Bayesian Sports Analytics

**Difficulte** : 3/5 | **Domaine** : Probas

**Description** :
Predire les resultats sportifs mieux que les bookmakers en modelisant la force des equipes (capacite d'attaque, solidite defensive) dans un championnat. Les modeles hierarchiques bayesiens capturent l'avantage du terrain, la variabilite entre equipes et l'incertitude sur les parametres. Application a un championnat reel (Ligue 1, Premier League, NBA).

**Objectifs gradues** :
- **Minimum** : Modele Poisson bayesien pour buts par match (attaque/defense par equipe), estimation sur une saison
- **Bon** : Modele hierarchique sous Stan/PyMC avec avantage terrain, prediction de resultats futurs avec intervalles
- **Excellent** : Comparaison avec cotes de bookmakers, analyse de value bets, modele dynamique (force evoluant dans le temps)

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Infer-101 | Modeles probabilistes | [Infer-101](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer-101.ipynb) |
| Pyro_RSA | Programmation probabiliste Pyro | [Pyro_RSA](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Pyro_RSA_Hyperbole.ipynb) |

**References externes** :
- [Stan Case Studies - Sports](https://mc-stan.org/users/documentation/case-studies.html) - Modeles hierarchiques pour le sport
- [Baio & Blangiardo (2010)](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) - Modele bayesien pour la Serie A
- [CmdStanPy](https://mc-stan.org/cmdstanpy/) - Interface Python pour Stan

---

### I.3 - Hanabi AI - Cooperation et Theory of Mind

**Difficulte** : 4/5 | **Domaine** : GameTheory

**Description** :
Hanabi est un jeu de cartes cooperatif unique ou l'on voit les cartes des autres mais pas les siennes. Les joueurs doivent communiquer des indices limites pour coordonner leurs actions. L'agent doit modeliser ce que les autres savent ("Theory of Mind") et interpreter les indices comme des signaux implicites. Le Hanabi Challenge (DeepMind, 2019) est un benchmark reconnu pour l'IA cooperative.

**Objectifs gradues** :
- **Minimum** : Agent rule-based basique jouant a Hanabi, evaluation du score moyen
- **Bon** : Agent RL (Rainbow DQN ou PPO) entraine dans l'environnement Hanabi, comparaison avec rule-based
- **Excellent** : Agent avec Theory of Mind explicite, jeu avec humains via interface, analyse des conventions emergentes

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-17 | Multi-Agent RL | [GT-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL.ipynb) |
| GT-6 | Evolution et confiance | [GT-6](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-6-EvolutionTrust.ipynb) |

**References externes** :
- [Hanabi Learning Environment (DeepMind)](https://github.com/deepmind/hanabi-learning-environment) - Environnement officiel
- [The Hanabi Challenge (arXiv)](https://arxiv.org/abs/1902.00506) - Paper de reference
- [Other-Play (Hu et al., NeurIPS 2020)](https://arxiv.org/abs/2003.02979) - Zero-shot coordination

---

### I.4 - Rational Speech Acts (RSA) - Pragmatique du Langage

**Difficulte** : 3/5 | **Domaine** : Probas, ML

**Description** :
Le framework RSA (Rational Speech Acts) modelise locuteur et auditeur comme des agents bayesiens recursifs qui raisonnent l'un sur l'autre. Le notebook Pyro_RSA du cours couvre les bases (implicature, hyperbole). Le projet va bien au-dela : modeliser des phenomenes linguistiques complexes (ironie, metaphore, langage non-litteral), evaluer si les modeles RSA expliquent mieux le comportement humain que des modeles simples, et explorer l'interface avec les LLMs modernes. Comment un GPT genere-t-il des hyperboles ? Un modele RSA peut-il ameliorer la generation de langage naturel ? Ces questions sont a la frontiere de la recherche en NLP computationnel.

**Objectifs gradues** :
- **Minimum** : Implementer un modele RSA etendu (au-dela du notebook) couvrant au moins 2 phenomenes pragmatiques (implicature scalaire + hyperbole ou ironie). Comparer les predictions du modele avec des jugements humains (datasets existants).
- **Bon** : Modele RSA multi-tours (dialogue pragmatique), avec lexique appris (lexicon learning RSA). Evaluation quantitative sur un dataset de reference (e.g., PRAG-dataset, SarcasmCorpus). Analyse de l'impact des hyperparametres (rationalite alpha, depth of recursion, prior).
- **Excellent** : Interface RSA-LLM : utiliser un LLM comme prior semantique du modele RSA (remplacement du literal listener), ou evaluer si un LLM (GPT-4, Claude) produit des jugements pragmatiques coherents avec RSA. Comparaison formelle RSA vs LLM sur des benchmarks de non-literal language (metaphore, ironie, understatement). Publication potentielle.

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| Pyro_RSA | RSA et pragmatique avec Pyro (base de demarrage) | [Pyro_RSA](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Pyro_RSA_Hyperbole.ipynb) |

**References externes** :
- [probLang.org](https://www.problang.org/) - Cours interactif Stanford (chapitres: implicature, hyperbole, irony, non-literal language, lexical uncertainty)
- [Pyro RSA Tutorial](https://pyro.ai/examples/RSA-implicature.html) - Tutoriel officiel Pyro (point de depart technique)
- [Goodman & Frank (2016)](https://www.sciencedirect.com/science/article/pii/S136466131630122X) - Paper fondateur du framework RSA
- [Bergen et al. (2016)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154975) - RSA pour l'hyperbole (etend le modele de base)
- [Kao et al. (2014)](https://web.stanford.edu/~jurafsky/kao14metaphor.pdf) - RSA pour la metaphore
- [WebPPL](https://webppl.org/) - Langage probabiliste du site probLang (alternative a Pyro)
- [RSA with Neural Semantics (EMNLP 2023)](https://aclanthology.org/2023.findings-emnlp.869/) - Interface RSA + LLMs

---

### I.5 - Kidney Exchange - Optimisation Combinatoire Cooperative

**Difficulte** : 4/5 | **Domaine** : GameTheory

**Description** :
Organiser des chaines d'echanges croises de reins pour sauver le maximum de vies. Un donneur incompatible avec son receveur peut echanger avec un autre couple donneur-receveur dans la meme situation. Le probleme combine optimisation combinatoire (trouver les cycles et chaines optimaux dans un graphe de compatibilite) et theorie des jeux cooperatifs (incitations a participer, mecanismes equitables). Ce probleme a des implications reelles : les travaux de Tuomas Sandholm (CMU) sont utilises par les programmes d'echange de reins aux USA.

**Objectifs gradues** :
- **Minimum** : Modelisation du graphe de compatibilite, recherche de cycles optimaux (algorithme exact ou heuristique)
- **Bon** : Chaines avec altruistic donors, analyse des incitations a participer (jeux cooperatifs), simulations
- **Excellent** : Mecanismes equitables (axe Shapley), comparaison avec politiques reelles, scalabilite

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-15 | Jeux cooperatifs | [GT-15](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-15-CooperativeGames.ipynb) |
| GT-16 | Mechanism Design | [GT-16](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-16-MechanismDesign.ipynb) |

**References externes** :
- [Sandholm - Kidney Exchange (CMU)](http://www.cs.cmu.edu/~sandholm/) - Travaux de reference
- [OpenSpiel (DeepMind)](https://github.com/deepmind/open_spiel) - Framework de jeux
- [OR-Tools (Google)](https://developers.google.com/optimization) - Optimisation combinatoire

---

### I.6 - RL pour Controle de Jeux (Snake/Mario/CartPole)

**Difficulte** : 3/5 | **Domaine** : ML

**Description** :
Apprendre a un agent RL a jouer a un jeu video (Snake, Mario, Doom) ou a controler un systeme physique (pendule inverse, CartPole). C'est le sujet d'introduction classique au RL avec des resultats visuellement impressionnants et pedagogiquement riches. Le projet compare plusieurs algorithmes (PPO, DQN, SAC) et analyse l'apprentissage (reward curves, strategies emergentes, generalisation).

**Objectifs gradues** :
- **Minimum** : Agent DQN sur CartPole, courbes d'apprentissage, comparaison avec random policy
- **Bon** : Comparaison PPO vs DQN vs SAC sur 2+ environnements, hyperparameter tuning, analyse qualitative
- **Excellent** : Environnement complexe (Mario, Doom), transfert entre jeux, curriculum learning, demo interactive

**Notebooks de reference** :

| Notebook | Description | Lien |
|----------|-------------|------|
| GT-17 | Multi-Agent RL | [GT-17](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL.ipynb) |

**References externes** :
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - Algorithmes RL de reference
- [SB3 Tutorials](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html) - Exemples complets (CartPole, Atari, custom envs)
- [Gymnasium](https://gymnasium.farama.org/) - Environnements RL standard (successeur OpenAI Gym)
- [Gymnasium Tutorials](https://gymnasium.farama.org/tutorials/training_agents/) - Tutoriels d'entrainement d'agents
- [RL Baselines Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - Hyperparametres optimises pour 100+ envs
- [Spinning Up (OpenAI)](https://spinningup.openai.com/) - Cours d'introduction au RL (theorie + code)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Implementations single-file de PPO, DQN, SAC (pedagogique)

[Retour a l'index des sujets](#index-des-sujets)

---

## Ressources Generales

### Outils et Plateformes

| Ressource | Description | Lien |
|-----------|-------------|------|
| **QuantConnect** | Plateforme de backtesting algorithmique (gratuite pour la recherche) | [quantconnect.com](https://www.quantconnect.com/) |
| **HuggingFace** | Modeles et datasets (NLP, CV, Audio) | [huggingface.co](https://huggingface.co/) |
| **Kaggle** | Datasets propres et notebooks d'exemple | [kaggle.com](https://www.kaggle.com/) |
| **PapersWithCode** | Etat de l'art par tache | [paperswithcode.com](https://paperswithcode.com/) |
| **ArXiv** | Papiers de recherche originaux | [arxiv.org](https://arxiv.org/) |
| **Yahoo Finance / yfinance** | Donnees financieres historiques gratuites | [pypi.org/project/yfinance/](https://pypi.org/project/yfinance/) |
| **Google Colab** | Jupyter notebooks avec GPU gratuit | [colab.google](https://colab.research.google.com/) |

### Notebooks du Cours (CoursIA)

Le depot [CoursIA](https://github.com/jsboige/CoursIA) contient l'ensemble des notebooks de reference :

| Serie | Notebooks | Themes |
|-------|-----------|--------|
| **Probas/Infer** | Infer-1 a Infer-20 + Infer-101 | Infer.NET, inference bayesienne, decision |
| **Probas/Pyro** | Pyro_RSA_Hyperbole | Programmation probabiliste, RSA |
| **GameTheory** | GT-1 a GT-17 | Nash, evolution, CFR, cooperation, mechanism design, MARL |
| **ML/ML.Net** | ML-1 a ML-7 | Introduction ML, features, entrainement, evaluation, time series |
| **ML/DataScience** | Lab1 a Lab17 | Agents data science, RAG, NLP, Kaggle |
| **QuantConnect** | QC-Py-01 a QC-Py-27 | Setup, data, strategies, ML, DL, RL, LLM, production |

### IA Generative

Vous etes **encourages a utiliser l'IA generative** (ChatGPT, Claude, GitHub Copilot, etc.) pour vous aider dans votre projet. L'ambition attendue reflete cette capacite : un projet realise avec assistance IA doit aller plus loin qu'un projet sans.

---

## Tendances IA et Finance 2025-2026

Pour vous aider dans vos choix, voici les 10 tendances majeures qui dominent le paysage IA/Finance en 2025-2026 :

| # | Tendance | Technologies cles | Sujets connexes |
|---|---------|-------------------|-----------------|
| 1 | **Agents IA et Multi-Agent Systems** | LangGraph, AutoGen, CrewAI | F.1, B.3, G.3 |
| 2 | **RAG (Retrieval Augmented Generation) Avance** | LlamaIndex, LangChain, Chroma | C.2, C.4 |
| 3 | **Neurosymbolic AI** | DeepProbLog, Neural Theorem Provers | F.5, D.3 |
| 4 | **Causal Inference et Discovery** | DoWhy, EconML, NOTEARS | G.1, A.5 |
| 5 | **Conformal Prediction et UQ** | MAPIE, Conformalized QR | A.5, A.6 |
| 6 | **LLMs pour Code Generation** | GPT-4, Claude, Copilot | Tous les sujets |
| 7 | **Time-Series Foundation Models** | Kronos, Chronos-2, Moirai | E.1, E.4, C.3 |
| 8 | **Differential Privacy et FL** | PySyft, Flower, OpenDP | D.1, D.2 |
| 9 | **Generative AI pour Stress Testing** | Diffusion, LLMs, GFlowNets | E.3, F.4, G.6 |
| 10 | **World Models et Model-Based RL** | DreamerV3, PINNs | G.4, G.5, E.6 |

### Conseils pour choisir votre sujet

1. **Alignement avec votre parcours** : Probas (cat. A, I.1-I.4), GameTheory (cat. B, I.3, I.5), ML (cat. C-H, I.6)
2. **Originalite** : Les sujets de recherche (cat. E, G) sont les plus originaux mais plus risques
3. **Employabilite** : RAG, Agents, Federated Learning, QuantConnect sont tres demandes en 2025-2026
4. **Ambition realiste** : Visez le niveau "Bon" minimum, "Excellent" si equipe solide et assistance IA
5. **Finance vs Non-Finance** : La categorie I offre des alternatives pour les passionnes de jeux, sport ou sante
