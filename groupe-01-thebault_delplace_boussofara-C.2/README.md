# BrightVest Financial RAG

Projet de groupe `groupe-01-thebault_delplace_boussofara-C.2`.

Ce dossier est le livrable autonome a remettre. Tout ce qui est necessaire au projet se trouve dedans.

Ce projet implemente un systeme RAG financier capable de repondre a des questions complexes sur des actions, des fondamentaux d'entreprise et le contexte macroeconomique. La base de connaissances provient de l'infrastructure Supabase de BrightVest, avec enrichissement en temps reel pour les prix et indicateurs techniques.

## Contexte

- Projet scolaire ECE ING4
- Objectif: construire un assistant financier base sur RAG
- Cible: demonstration academique et base reutilisable pour BrightVest
- Modele LLM: Groq `llama-3.3-70b-versatile`
- Embeddings: Ollama `nomic-embed-text`

## Fonctionnalites

- Ingestion de news, fondamentaux et indicateurs macro dans ChromaDB
- Retrieval hybride: dense + BM25 + reranking
- Reponses en mode `simple` et `analyst`
- API backend `FastAPI`
- Interface utilisateur `Next.js 14`
- Evaluation via benchmarks RAGAS, retrieval et ablation

## Structure Du Dossier

```text
groupe-01-thebault_delplace_boussofara-C.2/
|-- README.md
|-- chroma_db/
|-- src/
|-- frontend/
|-- docs/
|-- slides/
|-- requirements.txt
|-- .env.example
```

Documentation disponible:

- Guide de lancement: [guide-lancement.md]
- Rapport technique: [rapport-technique.md]
- Slides: [slides/README.md]

## Installation

Depuis ce dossier lui-meme:

```bash
cd groupe-01-thebault_delplace_boussofara-C.2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pour le frontend:

```bash
cd frontend
npm install
cd ..
```

## Variables D'Environnement

Copier le fichier d'exemple puis completer les secrets:

```bash
cp .env.example .env.local
```

Variables attendues:

```env
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
GROQ_API_KEY=gsk_...
```

Option frontend:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Usage

1. Lancer Ollama:

```bash
ollama serve
ollama pull nomic-embed-text
```

2. Indexer les donnees:

```bash
python -m src.ingestion.pipeline
```

Note: le dossier `chroma_db/` est deja inclus pour permettre un demarrage plus direct. La reindexation reste possible si necessaire.

3. Lancer l'API:

```bash
uvicorn src.api:app --reload --port 8000
```

4. Lancer le frontend:

```bash
cd frontend
npm run dev
```

## Resultats Actuels

Verification realisee sur ce depot:

- Build frontend `Next.js`: OK
- Compilation Python `src/`: OK
- CLI ingestion: OK
- CLI evaluation: OK

Statistiques ChromaDB observees:

- `news`: `5623` documents
- `earnings`: `3615` documents
- `macro`: `190` documents

Exemples de questions de demo:

- `What is the latest news on NVDA?`
- `Analyze AAPL fundamentals for me`
- `Compare MSFT and GOOGL revenue growth`
- `What is the current macro outlook with VIX and rates?`

## Documentation Technique

Le detail technique est dans [docs]

- architecture et pipeline RAG
- guide de lancement
- rapport technique du projet

## Slides De Presentation



## Instructions De Soumission

- Forker le depot cible sur GitHub
- Soumettre ce dossier `groupe-01-thebault_delplace_boussofara-C.2/` comme livrable complet
- Ouvrir une Pull Request au plus tard le `28 mars 2026`
- Ajouter les slides avant la presentation du `30 mars 2026` au matin

## Membres

- Thebault
- Delplace
- Boussofara
