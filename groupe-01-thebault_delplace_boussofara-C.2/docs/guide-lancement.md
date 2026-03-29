# Guide De Lancement

Ce document est la version "rendu école" du projet: il explique comment relancer le backend, le frontend, l'indexation RAG et l'évaluation sans devoir relire tout le code.

## 1. Ce Que Contient Le Projet

- Backend: API `FastAPI` dans [src/api.py](/Users/alexis/Documents/ECE/ING4/IA finance/projetRAG/brightvest-rag/groupe-01-thebault_delplace_boussofara-C.2/src/api.py)
- Frontend: interface `Next.js 14` dans [frontend](/Users/alexis/Documents/ECE/ING4/IA finance/projetRAG/brightvest-rag/groupe-01-thebault_delplace_boussofara-C.2/frontend)
- Ingestion RAG: pipeline dans [src/ingestion/pipeline.py](/Users/alexis/Documents/ECE/ING4/IA finance/projetRAG/brightvest-rag/groupe-01-thebault_delplace_boussofara-C.2/src/ingestion/pipeline.py)
- Evaluation: benchmarks dans [src/evaluation/eval_ragas.py](/Users/alexis/Documents/ECE/ING4/IA finance/projetRAG/brightvest-rag/groupe-01-thebault_delplace_boussofara-C.2/src/evaluation/eval_ragas.py)

## 2. Prerequis

- Python `3.11+`
- Node.js `18+`
- Ollama installe localement
- Acces aux variables d'environnement Supabase et Groq

## 3. Variables D'Environnement

Le backend lit les variables depuis `.env` puis `.env.local`.

Le plus simple:

```bash
cp .env.example .env.local
```

Puis remplir ce fichier avec:

```env
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
GROQ_API_KEY=gsk_...
```

Pour le frontend, la variable `NEXT_PUBLIC_API_URL` est optionnelle car le code pointe deja vers `http://localhost:8000` par defaut.

Si tu veux l'expliciter:

```bash
printf "NEXT_PUBLIC_API_URL=http://localhost:8000\n" > frontend/.env.local
```

## 4. Installation

### Backend Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend Next.js

```bash
cd frontend
npm install
cd ..
```

## 5. Premier Demarrage

### Terminal 1 - Ollama

```bash
ollama serve
```

### Terminal 2 - modele d'embeddings

```bash
ollama pull nomic-embed-text
```

### Terminal 3 - indexation ChromaDB

Premiere execution:

```bash
source .venv/bin/activate
python -m src.ingestion.pipeline
```

Verification rapide sans reindexer:

```bash
source .venv/bin/activate
python -m src.ingestion.pipeline --stats-only
```

### Terminal 4 - backend FastAPI

```bash
source .venv/bin/activate
uvicorn src.api:app --reload --port 8000
```

### Terminal 5 - frontend Next.js

```bash
cd frontend
npm run dev
```

Interface web:

- Frontend: [http://localhost:3000](http://localhost:3000)
- API health: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- API stats: [http://localhost:8000/api/stats](http://localhost:8000/api/stats)

## 6. Commandes Utiles

```bash
# Reindexer seulement une partie
python -m src.ingestion.pipeline --sources news
python -m src.ingestion.pipeline --sources fundamentals macro

# Reindexation complete
python -m src.ingestion.pipeline --force-reindex

# Evaluation
python -m src.evaluation.eval_ragas --benchmark ragas --questions 5
python -m src.evaluation.eval_ragas --benchmark retrieval --questions 5
python -m src.evaluation.eval_ragas --benchmark ablation --questions 5
```

Les resultats d'evaluation sont enregistres dans [src/evaluation/results](/Users/alexis/Documents/ECE/ING4/IA finance/projetRAG/brightvest-rag/groupe-01-thebault_delplace_boussofara-C.2/src/evaluation/results) lorsqu'un benchmark est execute.

## 7. Questions De Demo

- `What is the latest news on NVDA?`
- `Analyze AAPL fundamentals for me`
- `Compare MSFT and GOOGL revenue growth`
- `What is the current macro outlook with VIX and rates?`

## 8. Verification Rapide Avant Rendu

Commandes verifiees localement dans ce depot:

- `python -m compileall src`
- `.venv/bin/python -m src.ingestion.pipeline --help`
- `.venv/bin/python -m src.ingestion.pipeline --stats-only`
- `.venv/bin/python -m src.evaluation.eval_ragas --help`
- `npm run build` dans [frontend](/Users/alexis/Documents/ECE/ING4/IA finance/projetRAG/brightvest-rag/groupe-01-thebault_delplace_boussofara-C.2/frontend)

Etat observe pendant cette verification:

- Collection `news`: `5623` documents
- Collection `earnings`: `3615` documents
- Collection `macro`: `190` documents

## 9. Depannage

- Si `ModuleNotFoundError` apparait: activer `.venv` puis relancer `pip install -r requirements.txt`
- Si Ollama n'est pas joignable: lancer `ollama serve`
- Si l'API renvoie `503`: relancer `python -m src.ingestion.pipeline`
- Si le frontend affiche `API Offline`: verifier que `uvicorn` tourne bien sur le port `8000`
- Si l'evaluation RAGAS est lente: commencer avec `--questions 3` ou `--questions 5`
