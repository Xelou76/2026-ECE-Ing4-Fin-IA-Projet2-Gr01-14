# Rapport technique — BrightVest Financial RAG

**ECE Paris — ING4 — Projet Intelligence Artificielle appliquée à la finance**
**Date : Mars 2026**

---

## Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
   - 1.4 Sources hétérogènes : pourquoi pas de PDF ?
   - 1.5 Frameworks RAG : LangChain et LlamaIndex
2. [Architecture globale](#2-architecture-globale)
3. [Sources de données](#3-sources-de-données)
4. [Pipeline d'ingestion](#4-pipeline-dingestion)
5. [Pipeline de retrieval](#5-pipeline-de-retrieval)
6. [Pipeline de génération](#6-pipeline-de-génération)
7. [Fonctionnalités avancées](#7-fonctionnalités-avancées)
8. [API et interface utilisateur](#8-api-et-interface-utilisateur)
9. [Évaluation](#9-évaluation)
10. [Choix techniques et contraintes](#10-choix-techniques-et-contraintes)
11. [Limites et pistes d'amélioration](#11-limites-et-pistes-damélioration)

---

## 1. Contexte et objectifs

### 1.1 Problématique

Les outils financiers classiques répondent à des questions factuelles simples (cours d'une action, bénéfice annuel) mais échouent sur des questions complexes nécessitant la synthèse de plusieurs sources hétérogènes : actualités récentes, résultats fondamentaux, indicateurs macroéconomiques et données de marché en temps réel.

**Exemples de questions cibles :**
- *"Quelles sont les dernières nouvelles sur NVDA et quel est le signal directionnel ?"*
- *"Compare la croissance du chiffre d'affaires de MSFT et GOOGL sur les deux dernières années."*
- *"L'environnement macro actuel (VIX, taux) est-il favorable aux actions technologiques ?"*
- *"Développe le bear case de mon analyse ASML précédente."*

### 1.2 Approche RAG

La Retrieval-Augmented Generation (RAG) est une architecture qui augmente un LLM (Large Language Model) avec une base de connaissances externe interrogée dynamiquement. Au lieu de se fier uniquement aux paramètres du modèle (connaissance statique, soumise à une date de coupure), le système récupère des documents pertinents à chaque requête et les injecte dans le contexte du LLM.

**Avantages dans le domaine financier :**
- Réponses ancrées sur des données réelles et datées
- Traçabilité : chaque affirmation cite sa source
- Mise à jour sans ré-entraînement (ingestion des nouvelles données)
- Réduction des hallucinations grâce à la contrainte "contexte uniquement"

### 1.3 Périmètre du projet

Le système est construit au-dessus de l'infrastructure Supabase existante de BrightVest, qui agrège quotidiennement des données financières via des crons (Finnhub, RSS, FMP, Alpha Vantage, FRED). L'objectif est double : produire un projet académique évaluable (notebook RAGAS) et fournir un module prêt pour l'intégration en production dans la plateforme BrightVest.

### 1.4 Sources hétérogènes : pourquoi pas de PDF ?

Les énoncés de projets RAG citent classiquement les PDF comme exemple de source hétérogène (rapports annuels, prospectus, notes d'analystes). Ce système **n'implémente pas de parsing PDF**, et ce choix est délibéré.

BrightVest dispose d'une infrastructure Supabase déjà opérationnelle qui normalise toutes les données financières à la source : les articles de presse arrivent via Finnhub/RSS avec titre et résumé structurés, les fondamentaux viennent de FMP et Alpha Vantage sous forme de JSON typé, les indicateurs macro sont injectés depuis FRED. **Les données sont donc déjà structurées en base avant même d'atteindre le RAG.**

Parser des PDF de rapports annuels introduirait trois problèmes sans gain réel dans ce contexte :
1. **Qualité d'extraction** — les PDF financiers contiennent des tableaux, graphiques et layouts multi-colonnes que les extracteurs (PyMuPDF, pdfplumber) restituent imparfaitement.
2. **Redondance** — les chiffres des rapports annuels sont déjà présents dans `fundamentals_serving`, plus fiables car sourcés directement via API FMP/Alpha Vantage.
3. **Maintenance** — chaque émetteur a son propre format PDF ; l'extraction fragiliserait le pipeline.

Le choix d'une source unique et structurée (Supabase) est une décision d'architecture justifiée par le contexte BrightVest, pas une limitation technique.

### 1.5 Frameworks RAG : LangChain et LlamaIndex

LangChain et LlamaIndex sont les deux frameworks RAG de référence. Ils fournissent des abstractions prêtes à l'emploi (loaders, splitters, chains, agents). Ce projet **les utilise marginalement** (LangChain uniquement pour le wrapper `OllamaEmbeddings` et `ChatGroq`) et a construit le reste **from scratch**.

Ce choix est là aussi délibéré :

| Aspect | Framework (LangChain/LlamaIndex) | From scratch |
|--------|----------------------------------|--------------|
| Productivité initiale | Rapide à démarrer | Plus long |
| Contrôle fin | Abstractions rigides | Contrôle total |
| Debugging | Difficile (chaînes opaques) | Directement lisible |
| Adaptation domaine | Générique | Optimisable pour la finance |
| Dépendances | Lourdes, breaking changes fréquents | Légères et stables |

La fusion RRF, le routing agentique par query type, la gestion des tickers EU, le rate limiting Groq, la vérification des citations — tous ces composants auraient nécessité des contournements importants avec LangChain. Les construire directement a produit un code plus maintenable, plus lisible, et adapté précisément au domaine financier.

---

## 2. Architecture globale

### 2.1 Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                        DONNÉES (Supabase)                        │
│   articles · fundamentals_serving · macro_indicators            │
│   prices_daily · technical_indicators · positions               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Ingestion (offline)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR STORE (ChromaDB)                       │
│         news (5 623) · earnings (3 615) · macro (190)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────────┐
          │                 │                       │
          ▼                 ▼                       ▼
┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐
│ Dense (ANN)  │  │  Sparse (BM25)   │  │  Live Enrichment      │
│ ChromaDB +   │  │  rank-bm25       │  │  prix · technicals    │
│ nomic-embed  │  │  in-memory       │  │  (Supabase on demand) │
└──────┬───────┘  └────────┬─────────┘  └───────────┬───────────┘
       │                   │                         │
       └─────────┬─────────┘                         │
                 ▼                                    │
       ┌──────────────────┐                          │
       │   Hybrid RRF     │                          │
       │  (fusion ranks)  │                          │
       └────────┬─────────┘                          │
                ▼                                    │
       ┌──────────────────┐                          │
       │  Cross-encoder   │                          │
       │   Reranker       │                          │
       └────────┬─────────┘                          │
                │                                    │
                └──────────────┬─────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   RAG Generator     │
                    │   Groq LLM          │
                    │   (llama-3.3-70b)   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  SimpleAnswer  /    │
                    │  AnalystAnswer      │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  FastAPI · Next.js  │
                    └─────────────────────┘
```

### 2.2 Structure du code

```
src/
├── config.py                   # Configuration centrale (env vars, hyperparamètres)
├── api.py                      # Endpoints FastAPI
├── data/
│   └── supabase_client.py      # Client Supabase read-only, toutes les requêtes typées
├── ingestion/
│   ├── pipeline.py             # Orchestrateur (CLI + programmatique)
│   ├── news_indexer.py         # articles → chunks → ChromaDB "news"
│   ├── fundamentals_indexer.py # fundamentals_serving → ChromaDB "earnings"
│   └── macro_indexer.py        # macro_indicators → ChromaDB "macro"
├── retrieval/
│   ├── dense.py                # DenseRetriever — ANN via ChromaDB
│   ├── sparse.py               # SparseRetriever — BM25 via rank-bm25
│   ├── hybrid.py               # HybridRetriever — fusion RRF
│   ├── reranker.py             # Reranker — cross-encoder ms-marco
│   └── context_builder.py      # ContextBuilder — enrichissement live
├── generation/
│   ├── agent.py                # FinancialAgent — planner agentique
│   ├── generator.py            # RAGGenerator — appels LLM Groq
│   └── prompts.py              # Tous les templates de prompts
└── evaluation/
    └── eval_ragas.py           # Benchmarks RAGAS + ablation
```

---

## 3. Sources de données

### 3.1 Données indexées dans ChromaDB

#### Articles de presse — collection `news`

| Champ | Description |
|-------|-------------|
| `ticker` | Symbole boursier (US ou EU avec suffixe) |
| `headline` | Titre de l'article |
| `summary` | Résumé (quelques phrases) |
| `source` | Origine (Finnhub 87%, RSS 11%, legacy 2%) |
| `published_at` | Date de publication (ISO-8601) |
| `sector` | Secteur d'activité |
| `sentiment_final` | Sentiment (souvent vide) |
| `category` | Catégorie éditoriale |
| `region` | Région (US / EU) |
| `dedup_hash` | Hash de déduplication |

**Volume :** 52 854 articles en base, 5 623 indexés (5 000 US les plus récents + 623 EU dédiés).

#### Fondamentaux — collection `earnings`

Données financières par ticker et par période (annuelle/trimestrielle) :
- **Compte de résultat** : revenue, gross_profit, operating_income, net_income, ebitda, eps
- **Marges** : gross_margin, operating_margin, net_margin
- **Bilan** : total_assets, total_liabilities, total_equity, cash, total_debt
- **Cash flows** : operating_cash_flow, capex, free_cash_flow
- **Ratios de valorisation** : pe_ratio_ttm, peg_ratio_ttm, ev_to_ebitda_ttm, price_to_book_ttm
- **Rentabilité** : roe, roa, roic_ttm
- **Croissance** : revenue_growth_yoy, net_income_growth_yoy

**Volume :** 3 701 lignes (~739 tickers). Sources : FMP (989), Alpha Vantage (2 626), legacy (86).

#### Indicateurs macro — collection `macro`

| Série | Description | Observations |
|-------|-------------|-------------|
| VIX | Indice de volatilité implicite S&P 500 | 9 108 (historique long) |
| FEDFUNDS | Taux directeur Fed Funds | 553 |
| DGS10 | Rendement Treasury 10 ans | ~21 |
| DGS2 | Rendement Treasury 2 ans | ~21 |
| T10Y2Y | Spread 10ans–2ans (courbe des taux) | ~21 |
| ICSA | Inscriptions chômage hebdomadaires | 4 |
| VIXCLS | VIX clôture (FRED) | 22 |

**Volume indexé :** 190 chunks (plafonné à 50 valeurs par série pour équilibrer la représentation).

### 3.2 Données live (non indexées, récupérées à la volée)

#### Prix journaliers — `prices_daily`
- 451 110 lignes · OHLCV + adj_close
- Utilisé pour : évolution récente d'un titre, calcul de performances sur période custom

#### Indicateurs techniques — `technical_indicators`
- 424 275 lignes
- Returns : 1j, 5j, 20j
- Momentum : RSI(14), MACD, signal, histogramme
- Risque : volatilité 20j, drawdown max 1an glissant
- Volume : moyenne 20j, turnover ratio

#### Positions portefeuille — `portfolios` / `positions`
- 4 portefeuilles, 7 positions
- Utilisé pour l'analyse de portefeuille personnalisée

---

## 4. Pipeline d'ingestion

### 4.1 Orchestration

Le pipeline est contrôlé par `IngestionPipeline` qui séquence les trois indexeurs. Chaque source est encapsulée dans un `try/except` pour qu'un échec n'interrompe pas les autres.

```bash
# CLI
python -m src.ingestion.pipeline                    # tout indexer
python -m src.ingestion.pipeline --sources news     # news uniquement
python -m src.ingestion.pipeline --force-reindex    # tout reconstruire
python -m src.ingestion.pipeline --stats-only       # stats sans indexer
```

### 4.2 Stratégie de chunking

#### News
Un chunk = un article. Contenu : `"{headline}. {summary}"`.

La déduplication utilise le `dedup_hash` de Supabase comme identifiant ChromaDB — les articles déjà présents sont automatiquement ignorés par l'upsert.

**Cas particulier EU :** les articles européens datent de 2021-2022 et sont donc hors de la fenêtre temporelle des 5 000 articles les plus récents. Une méthode dédiée `index_eu_articles()` les récupère via un filtre `region='EU'` indépendamment de la limite.

#### Fondamentaux
Un chunk = un (ticker, période). Les valeurs numériques sont converties en langage naturel pour être compatibles avec la recherche sémantique :

```
"AAPL annual results ending 2024-09-28:
Revenue: $391,035,000,000, Net income: $93,736,000,000, EPS: $6.11.
Margins: gross 46.2%, operating 31.5%, net 24.0%.
Growth: revenue YoY +2.0%.
Valuation: P/E 33.4, EV/EBITDA 26.1.
Balance sheet: debt/equity 1.87, FCF: $108,807,000,000.
Market cap: $3,452,000,000,000, Beta: 1.24."
```

Les champs NULL sont omis proprement. Un mécanisme de déduplication par `doc_id` dans le batch évite les erreurs ChromaDB sur les lignes avec `fiscal_date_ending = NULL`.

#### Macro
Un chunk = une valeur d'une série à une date. Le plafonnement à 50 valeurs par série empêche VIX (9 108 points) d'écraser les autres séries dans la collection.

### 4.3 Embeddings

Modèle : **`nomic-embed-text`** via Ollama (local, Apple Silicon).
- Dimension : 768
- Distance ChromaDB : cosinus (`hnsw:space: cosine`)
- Batch size : 100 (news/macro), 50 (fundamentals)
- Entièrement local, aucun coût d'API

---

## 5. Pipeline de retrieval

### 5.1 Retrieval dense (ANN)

`DenseRetriever` embarque la requête via Ollama REST API puis exécute une recherche approximate nearest-neighbour dans ChromaDB.

**Paramètres :**
- `DENSE_TOP_K = 15` documents retournés
- Filtres metadata supportés : `ticker` (avec variantes de suffixes EU), `doc_type`, `date_from`

**Gestion des tickers européens :**
ChromaDB stocke `ASML.AS`, `LVMH.PA`, `SAP.DE` mais l'utilisateur tape `ASML`. Le filtre dense utilise `$in` avec 16 variantes de suffixes (`.L`, `.PA`, `.DE`, `.AS`, `.MI`, `.SW`, `.BR`, `.MC`, `.HE`, `.OL`, `.ST`, `.CO`, `.LS`, `.WA`, `.BU`, `.AT`).

### 5.2 Retrieval sparse (BM25)

`SparseRetriever` charge toute la collection en mémoire au démarrage et construit un index BM25Okapi via la bibliothèque `rank-bm25`.

**Tokenisation spécialisée finance :**
- Préserve `$`, `%`, `.`, `-`, `_` (ticker symbols, valeurs monétaires)
- Conserve les chiffres (`2.1B`, `P/E 35.2`)
- Supprime les ponctuations parasites

**Avantage vs dense :** le BM25 excelle sur les termes exacts : symboles (`NVDA`, `AAPL`), métriques (`EBITDA`, `EPS`), séries macro (`FEDFUNDS`, `DGS10`), valeurs numériques.

**Gestion tickers EU côté sparse :** filtre post-scoring avec `startswith(ticker + ".")` pour matcher `ASML.AS` quand on cherche `ASML`.

### 5.3 Fusion hybride (RRF)

`HybridRetriever` combine les deux retrievers par **Reciprocal Rank Fusion** :

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
              i∈{dense, sparse}
```

avec `k = 60` (constante standard RRF).

Les deux appels (dense + sparse) sont exécutés en parallèle via `ThreadPoolExecutor` pour minimiser la latence. `HYBRID_TOP_K = 30` candidats sont passés au reranker.

**Pourquoi le hybride ?**
- Dense seul rate les requêtes à termes exacts (ticker mal connu, valeur numérique précise)
- Sparse seul rate les requêtes sémantiques (paraphrases, synonymes)
- La fusion combine les avantages des deux avec un overhead minimal

### 5.4 Reranking cross-encoder

`Reranker` utilise `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`.

Contrairement aux bi-encodeurs (dense), un cross-encoder analyse la paire (requête, document) conjointement, ce qui permet une modélisation plus fine de la pertinence. En contrepartie, il ne peut pas être pré-calculé — il ne s'applique que sur les `HYBRID_TOP_K = 30` candidats pré-filtrés.

`RERANK_TOP_K = 5` documents finaux sont transmis au LLM.

**Point important :** `ms-marco-MiniLM-L-6-v2` a été entraîné sur des passages web (MS MARCO). Les textes financiers produisent des scores systématiquement négatifs (-6 à -9) même lorsqu'ils sont très pertinents. Le seuil absolu de score est donc inutilisable — seul le classement relatif est exploité.

### 5.5 Routing agentique

`FinancialAgent` orchestre tout le retrieval en deux étapes : **plan** puis **retrieve**.

#### Détection des tickers
Regex `\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b` suivie d'une liste d'exclusions (mots anglais courants : `I`, `A`, `THE`, `AND`, `FOR`, `ARE`, `NOT`…).

#### Classification du type de requête

| Type | Mots-clés déclencheurs | Collections ciblées |
|------|----------------------|---------------------|
| `news` | news, article, report, announced… | news |
| `earnings` | earnings, revenue, profit, EPS, margin… | earnings |
| `comparison` | compare, vs, versus, against, better… | earnings + news |
| `price` | price, drop, rise, rally, crash, trading… | prices_daily + technical_indicators |
| `momentum` | momentum, trend, moving average, breakout, EMA… | technical_indicators + news |
| `macro` | macro, fed, inflation, rates, VIX, GDP… | macro + news |
| `portfolio` | portfolio, positions, holdings, allocation… | positions + news + earnings |
| `recommendation` | buy, sell, invest, recommend, target… | earnings + news + technical |

#### Génération de sous-requêtes

Pour les requêtes complexes, l'agent génère plusieurs sous-requêtes spécialisées :

- **Macro :** expansion des séries mentionnées (`"federal reserve" → ["FEDFUNDS federal funds rate", "interest rate monetary policy"]`)
- **Comparison :** sous-requête par ticker (`"Compare MSFT GOOGL revenue" → ["MSFT revenue growth earnings", "GOOGL revenue growth earnings"]`)
- **Earnings :** variantes temporelles (`"AAPL earnings" → ["AAPL annual results", "AAPL quarterly results", "AAPL revenue growth"]`)

Déduplication systématique des sous-requêtes identiques pour éviter les appels redondants.

---

## 6. Pipeline de génération

### 6.1 LLM et rate limiting

**Modèle :** `llama-3.3-70b-versatile` via Groq API
- Température : `0.1` (réponses factuelles, peu créatives)
- Limite Groq free tier : 30 req/min, 14 400/jour
- Rate limiter intégré : `2s` minimum entre deux appels consécutifs
- Retry automatique en cas de `429` (jusqu'à 3 tentatives avec backoff)

### 6.2 Modes de réponse

#### Mode Simple (`SimpleAnswer`)
```json
{
  "answer": "Texte de réponse avec citations [Source N]",
  "sources": [
    {
      "type": "news",
      "ticker": "NVDA",
      "date": "2026-03-15T10:30:00",
      "detail": "NVIDIA reports record Q4 revenue...",
      "verified": true,
      "badge": "✅"
    }
  ],
  "confidence": "high"
}
```

Le prompt système contraint le LLM à répondre **uniquement** depuis le contexte fourni et à citer chaque fait avec `[Source N]`.

#### Mode Analyst (`AnalystAnswer`)
```json
{
  "answer": "Résumé exécutif 2-3 phrases",
  "bull_case": "Arguments haussiers ancrés dans le contexte",
  "bear_case": "Arguments baissiers et risques",
  "risks": ["Risque 1", "Risque 2"],
  "catalysts": ["Catalyseur 1", "Catalyseur 2"],
  "key_metrics": {
    "revenue_growth_yoy": "+122% [Source 3]",
    "pe_ratio": "35.2 [Source 1]",
    "rsi_14": "62.5 (neutre)"
  },
  "confidence": "high",
  "signal": "bullish"
}
```

Le LLM produit du JSON. Un fallback regex (`\{[\s\S]*\}`) extrait le JSON même quand le modèle omet les balises markdown.

### 6.3 Vérification des citations

Après génération, les sources sont marquées `verified=True` uniquement si le LLM a effectivement utilisé le `[Source N]` correspondant dans sa réponse. Les sources non citées reçoivent `verified=False` et un badge `⚠️`.

### 6.4 Inférence de confiance

| Condition | Confiance |
|-----------|-----------|
| ≥ 4 sources vérifiées | `high` |
| ≥ 2 sources vérifiées | `medium` |
| 1 source vérifiée | `low` |
| 0 sources vérifiées (LLM sans `[Source N]`) | `medium` si ≥ 2 docs, sinon `low` |

Ce dernier cas plafonne à `medium` car on ne peut pas vérifier quelles sources ont été utilisées.

### 6.5 Prompts

Tous les prompts sont centralisés dans `prompts.py` :

| Prompt | Usage |
|--------|-------|
| `QA_SYSTEM_PROMPT` | Instructions système mode simple |
| `QA_USER_TEMPLATE` | Template utilisateur mode simple |
| `ANALYST_SYSTEM_PROMPT` | Instructions système mode analyst |
| `ANALYST_USER_TEMPLATE` | Template utilisateur mode analyst (JSON forcé) |
| `QUERY_REWRITE_TEMPLATE` | Réécriture de requête pour améliorer le retrieval |
| `STANDALONE_QUESTION_TEMPLATE` | Contextualisation des questions de suivi |

---

## 7. Fonctionnalités avancées

### 7.1 Enrichissement live

`ContextBuilder` injecte des données temps-réel de Supabase **à la volée**, sans passer par ChromaDB.

**Déclenchement :** quand des tickers sont détectés et que le type de requête est `price`, `technical`, `momentum` ou `recommendation`.

**Données injectées :**

*Prix récents (30 derniers jours) :*
```
NVDA — Last 30 days price data:
Latest close: $875.40 (2026-03-19)
30-day range: $802.10 – $912.30
30-day return: +8.9%
Average volume: 45,230,000
```

*Indicateurs techniques :*
```
NVDA — Latest Technical Indicators (2026-03-19):
RSI(14): 62.5 (neutral)
MACD: 12.3, Signal: 8.7, Histogram: 3.6 (bullish momentum)
Returns: 1d: +1.2%, 5d: +4.8%, 20d: +8.9%
Volatility(20d): 28.5%
Max Drawdown(1y): -35.2%
```

RSI est interprété automatiquement : `>70` → "overbought", `<30` → "oversold", entre les deux → "neutral".

### 7.2 Historique de conversation

Le système supporte les questions de suivi grâce à deux mécanismes :

**Étape 1 — Contextualisation pour le retrieval :**
Avant d'interroger ChromaDB, la question de suivi est réécrite en question autonome via un appel LLM dédié.

```
Historique :
User: Analyse NVDA
Assistant: NVDA affiche une croissance revenue +122%...

Question : "Développe le bear case"
→ Réécriture : "Développe le bear case de NVDA : risques de valorisation
  élevée, concurrence AMD et Intel, dépendance aux datacenters"
```

Cela permet au retrieval de trouver les documents pertinents même avec une question elliptique.

**Étape 2 — Historique dans le prompt de génération :**
Les 3 derniers tours (6 messages, tronqués à 500 chars) sont injectés en tête du prompt LLM sous forme :
```
Conversation History:
User: Analyse NVDA
Assistant: NVDA affiche une forte croissance revenue...

[contexte RAG]

Question: Développe le bear case
```

Le LLM peut ainsi référencer et approfondir ses réponses précédentes.

### 7.3 Support des marchés européens

Les tickers européens utilisent des suffixes d'exchange non standards (ex: `ASML.AS` pour Amsterdam, `LVMH.PA` pour Paris, `SAP.DE` pour Frankfurt). Le système gère cette particularité à trois niveaux :

- **Ingestion :** `index_eu_articles()` contourne la limite temporelle des 5 000 articles US via un filtre `region='EU'` direct
- **Dense retrieval :** filtre `$in` avec 16 variantes de suffixes pour chaque ticker base
- **Sparse retrieval :** fallback `startswith(ticker + ".")` dans le filtre post-scoring

### 7.4 Analyse de portefeuille

`answer_portfolio()` personnalise l'analyse en fonction des positions réelles de l'utilisateur :

1. Récupère les positions Supabase (`portfolios` → `positions`)
2. Extrait les tickers
3. Construit un résumé des positions (quantité, PRU, devise)
4. Injecte tickers + résumé dans la question avant de lancer le pipeline RAG

```python
gen.answer_portfolio("Analyse mon portefeuille", user_id="uuid-123", mode="analyst")
```

### 7.5 Intégration BrightVest — `get_financial_signal()`

Point d'entrée pour la production :

```python
signal = get_financial_signal("NVDA")
# {
#   "ticker": "NVDA",
#   "signal": "bullish",
#   "confidence": 0.85,
#   "bull_case": "...",
#   "bear_case": "...",
#   "key_metrics": {"revenue_growth_yoy": "+122%", ...},
#   "news_count": 45,
#   "date_range": "2026-02-17 to 2026-03-19",
#   "error": null
# }
```

---

## 8. API et interface utilisateur

### 8.1 Backend FastAPI

**Endpoints :**

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/api/chat` | Question RAG (simple ou analyst) |
| `GET` | `/api/stats` | Nombre de documents par collection |
| `GET` | `/api/health` | État du backend |

**Initialisation :** `RAGGenerator` est instancié une seule fois au démarrage (`lifespan`), ChromaDB et BM25 sont donc chargés en mémoire. Si les collections n'existent pas (ingestion non faite), le backend démarre quand même et retourne un `503` clair.

**Modèle de requête :**
```json
{
  "question": "Analyse NVDA",
  "mode": "analyst",
  "ticker": "NVDA",
  "history": [
    {"role": "user", "content": "Latest NVDA news?"},
    {"role": "assistant", "content": "NVDA announced record revenue..."}
  ]
}
```

### 8.2 Frontend Next.js 14

Interface chat dark theme construite avec Tailwind CSS.

**Fonctionnalités UI :**
- Toggle Simple / Analyst
- Indicateur de statut backend (vert/rouge/jaune)
- Suggestions de questions au démarrage
- Spinner animé pendant le chargement
- Rendu Markdown des réponses
- Carte analyst structurée (bull/bear/métriques/sources/signal)
- Badge confiance (high/medium/low)
- Sources avec badge ✅/⚠️ (citées ou non par le LLM)
- Footer stats (nombre de docs indexés par collection)
- Historique de conversation persistant en session
- Auto-scroll, auto-resize textarea, Shift+Enter pour sauter une ligne

---

## 9. Évaluation

### 9.1 Métriques RAGAS

Le framework RAGAS évalue trois dimensions sans annotations humaines, en utilisant le LLM comme juge :

| Métrique | Description | Idéal |
|----------|-------------|-------|
| **Faithfulness** | Les affirmations de la réponse sont-elles supportées par le contexte récupéré ? | → 1.0 |
| **Answer Relevancy** | La réponse répond-elle à la question posée ? | → 1.0 |
| **Context Precision** | Les documents récupérés sont-ils pertinents pour la question ? | → 1.0 |

### 9.2 Benchmark de retrieval

Comparaison de trois configurations sur un jeu de questions de référence :

| Configuration | Avantage |
|--------------|----------|
| Dense seul | Requêtes sémantiques |
| Hybrid (dense + BM25) | Meilleur recall global |
| Hybrid + Rerank | Meilleure précision top-5 |

### 9.3 Ablation RAG vs LLM seul

Mesure la contribution du contexte RAG sur la qualité des réponses en comparant :
- Réponse LLM sans contexte (paramètres seuls)
- Réponse LLM avec contexte RAG (pipeline complet)

```bash
python -m src.evaluation.eval_ragas --benchmark ragas
python -m src.evaluation.eval_ragas --benchmark retrieval
python -m src.evaluation.eval_ragas --benchmark ablation
```

---

## 10. Choix techniques et contraintes

### 10.1 Pourquoi Groq + llama-3.3-70b ?

Groq propose un accès gratuit à llama-3.3-70b avec une inférence extrêmement rapide (~500 tokens/s). C'est le seul LLM 70B accessible gratuitement avec une vitesse suffisante pour une démo interactive. La contrainte est le rate limit (30 req/min) géré par le rate limiter intégré.

### 10.2 Pourquoi Ollama + nomic-embed-text ?

Entièrement local (aucun coût d'API), optimisé Apple Silicon (Metal GPU via llama.cpp), 768 dimensions suffisantes pour du texte financier court. L'alternative (OpenAI text-embedding-3-small) coûterait ~$0.02/1M tokens mais introduirait une dépendance externe.

### 10.3 Pourquoi ChromaDB ?

Base vectorielle légère, sans serveur à configurer, persistante sur disque. Adapté aux collections de taille modeste (< 100k documents). Pour des millions de documents, Qdrant ou Weaviate seraient plus appropriés.

### 10.4 Pourquoi BM25 en complément ?

Les embeddings sémantiques échouent sur les termes rares ou techniques exacts : `FEDFUNDS`, `T10Y2Y`, `EBITDA`, `P/E`, valeurs numériques précises. BM25 compense ces cas avec un overhead mémoire acceptable (corpus < 10k documents chargé en RAM).

### 10.5 Température 0.1

Choix délibéré pour maximiser la fiabilité factuelle. Le LLM doit citer des données réelles, pas improviser — une température basse réduit les variations créatives et les hallucinations.

### 10.6 Cross-encoder ms-marco-MiniLM

Modèle 22M paramètres, très rapide (< 100ms sur CPU pour 30 candidats). Entraîné sur des passages web (MS MARCO), ce qui explique les scores négatifs sur les textes financiers. Le reranking reste efficace car seul le classement relatif est utilisé.

---

## 11. Limites et pistes d'amélioration

### 11.1 Limites actuelles

**Volume de données indexées :**
Seuls 5 623 articles sur 52 854 sont indexés (limite arbitraire de 5 000 US + 623 EU). La contrainte est le coût en temps d'embedding Ollama local (~100 docs/min). Un serveur dédié ou une API d'embeddings accélèrerait massivement l'ingestion.

**Qualité du contenu articles :**
Les articles ne contiennent que le titre + résumé (pas le texte complet). La densité informationnelle par chunk est donc limitée.

**Sentiment fields vides :**
Le champ `sentiment_final` est absent pour la majorité des articles, ce qui empêche d'exploiter le sentiment comme signal de retrieval ou de génération.

**Données EU limitées :**
Les 728 articles EU datent de 2021-2022. Les données fondamentales EU (ASML, LVMH, SAP) sont disponibles mais les prix et technicals peuvent être absents pour certains tickers.

**Rate limit Groq :**
L'historique de conversation introduit un appel LLM supplémentaire (contextualisation), réduisant la capacité à 15 req/min en pratique lors de conversations soutenues.

### 11.2 Pistes d'amélioration

**Retrieval :**
- Indexer l'ensemble des 52 854 articles (nécessite un serveur d'embeddings plus rapide)
- Ajouter un filtre temporel natif (ChromaDB 2.x supporte `$gte` sur chaînes)
- Parent-child chunking pour les fondamentaux (chunk court pour retrieval, chunk long pour contexte)

**Génération :**
- Streaming des réponses (FastAPI `StreamingResponse` + `EventSource` côté frontend)
- Mémoire longue terme (résumer les N derniers tours pour économiser les tokens)
- Modèle spécialisé finance (FinBERT pour le reranking à la place de ms-marco)

**Données :**
- Ingestion full-text des articles (scraping ou API premium Finnhub)
- Alertes en temps réel (webhook Supabase → réindexation partielle)
- Couverture EU étendue : mise à jour quotidienne des articles et prix européens

**Évaluation :**
- Dataset de référence annoté manuellement (ground truth financier)
- Métriques métier : précision des signaux bull/bear sur l'historique
- A/B testing des stratégies de chunking et de retrieval

---

## Annexe — Hyperparamètres

| Paramètre | Valeur | Fichier |
|-----------|--------|---------|
| `EMBEDDING_MODEL` | `nomic-embed-text` | config.py |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | config.py |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | config.py |
| `LLM_TEMPERATURE` | `0.1` | config.py |
| `GROQ_RPM_LIMIT` | `30` | config.py |
| `NEWS_FETCH_LIMIT` | `5000` | config.py |
| `DENSE_TOP_K` | `15` | config.py |
| `HYBRID_TOP_K` | `30` | config.py |
| `RERANK_TOP_K` | `5` | config.py |
| `_EMBED_BATCH_SIZE` (news) | `100` | news_indexer.py |
| `_EMBED_BATCH_SIZE` (fundamentals) | `50` | fundamentals_indexer.py |
| `_VALUES_PER_SERIES` (macro) | `50` | macro_indexer.py |
| `RRF_k` | `60` | hybrid.py |
| `MAX_RETRIES` (Groq) | `3` | generator.py |
| `RETRY_WAIT_SECONDS` | `60` | generator.py |
| History window | `6 messages` | generator.py |
| History truncation | `500 chars/msg` | generator.py |

---

*Rapport généré le 20 mars 2026 — BrightVest Financial RAG v1.0*
