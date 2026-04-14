# Liquid_paper_fetch

Automated literature tracker for water science and crystallization topics.

## Features

- Multi-source fetch: OpenAlex, Crossref, Semantic Scholar.
- Configurable filtering: last N days, journal allowlist, embedding-based semantic reranking (with lexical fallback).
- AI processing: abstract translation (EN->ZH) and 1-2 sentence contribution summary.
- Structured Markdown report export.
- Downloadable release package (zip artifact) for local use.

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Edit `config.yaml` as needed.
3. Run:

```bash
python main.py
```

Output files will be generated under `reports/`.

## Full Configuration Guide

### 1) Time and Fetch Scale

- `general.lookback_days`: day window for publication date filtering.
- `general.max_records_per_source`: fetch volume per source.

### 2) Journal and Topic Filtering

- `filters.journals.issn`: ISSN allowlist.
- `filters.journals.titles`: journal title allowlist.
- `filters.keywords`: primary topic phrases.

### 3) Semantic Retrieval (Embedding Rerank)

`filters.semantic_search` options:

- `enabled`: whether semantic retrieval is enabled.
- `query_expansion`: enable synonyms/derivative query expansion.
- `model_name`: embedding model name.
- `similarity_threshold`: primary semantic score threshold.
- `top_k`: max kept papers after rerank.
- `min_hits`: minimum kept papers; if threshold hits are below this number, system auto-backfills from top semantic candidates.

Notes:

- The system auto-considers synonyms and derivative terms for core concepts such as ice nucleation, supercooled water, CNT, glass transition, fragile-to-strong transition, and amorphous/glassy water.
- If `sentence-transformers` is unavailable, the system automatically falls back to lexical keyword matching.

### 4) API Settings

- `apis.openalex.mailto`: optional recommended contact email.
- `apis.crossref.mailto`: optional recommended contact email.
- `apis.semanticscholar.api_key`: optional API key.
- `llm.api_key`: API key for translation/summarization endpoint.

## Release Package (Download to Local)

### Option A: Download from GitHub Releases

1. Open repository Releases page.
2. Download `Liquid_paper_fetch_<version>.zip`.
3. Unzip and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Option B: Build Local Release Package

```bash
python scripts/build_release.py --version v0.1.0
```

Generated artifact path:

- `dist/Liquid_paper_fetch_v0.1.0.zip`

## CI Release Workflow

Workflow file: `.github/workflows/release-package.yml`

- Trigger on tag push `v*` or manual dispatch.
- Automatically builds zip package.
- Uploads artifact in Actions run.
- If triggered by tag, publishes the zip into GitHub Release assets.

## Biweekly Automation

This repository includes a GitHub Actions workflow at `.github/workflows/biweekly-report.yml`.

- Trigger: weekly (Monday, UTC) + manual trigger.
- Biweekly logic: only runs on even ISO week numbers.
- Result: generated markdown report is committed into `reports/`.

If needed, configure these repository secrets:

- `SEMANTIC_SCHOLAR_API_KEY`
- `OPENROUTER_API_KEY`
