# CLAUDE.md — Stevenson

## Overview

Stevenson is an art discovery platform that scrapes painting listings from Craigslist and eBay, scores them with CLIP + LAION aesthetic models, and serves a gallery frontend backed by Redis.

## Architecture

**Frontend**: Static HTML pages served by Vercel
- `index.html` — Main gallery with infinite scroll, price filters, detail view
- `analytics.html` — Charts (Chart.js), style/artist breakdowns, hunter explorer
- `data.html` — Live data pipeline dashboard (polls `/api/pipeline` every 10s)
- `about.html` — About page

**API** (Vercel serverless, TypeScript):
- `api/paintings.ts` — Paginated painting queries with filters (price, region, sort)
- `api/stats.ts` — Aggregate stats (live scored count via ZCOUNT)
- `api/pipeline.ts` — Pipeline status: per-source scoring progress, coverage %

**Backend** (Python CLI + scripts):
- `stevenson` — CLI entrypoint (`pull`, `score`, `push`, `export`, `update`, `status`, `ingest`)
- `scorer.py` — Streaming CLIP scorer: chunks from stores → score → push to Redis → writeback
- `push.py` — Push paintings to Redis (`push_paintings()` + `push_stats()` reusable functions)
- `scraper.py` — Orchestrator for CL/eBay scrapers, store management, export
- `scrapers/craigslist.py` — Craigslist scraper (all US regions)
- `scrapers/ebay.py` — eBay scraper

**Data stores**: `data/craigslist.json`, `data/ebay.json` (gitignored, local-only)

**Redis** (Upstash/Redis Cloud):
- `stv:p:{id}` — Painting hash (all fields as strings)
- `stv:idx:art_score` — Sorted set for score-based queries
- `stv:idx:price` — Sorted set for price-based queries
- `stv:idx:value_score` — Sorted set for gem discovery
- `stv:idx:region:{name}`, `stv:idx:state:{code}`, `stv:idx:source:{cl|eb}` — Filtered indexes
- `stv:dedup` — Set of all painting IDs
- `stv:stats` — Hash of aggregate stats

## CLI Usage

```
stevenson pull                  # Scrape new listings (CL + eBay)
stevenson pull --cl --state ca  # Craigslist California only
stevenson score                 # Stream-score unscored paintings, push chunks to Redis
stevenson score --rescore       # Re-score everything
stevenson score --chunk-size 50 --limit 100  # Small test run
stevenson score --no-push       # Score without pushing to Redis
stevenson push                  # Push paintings.json to Redis
stevenson push --flush          # Flush + push
stevenson export                # Merge stores → paintings.json
stevenson update                # Full pipeline: pull → export → score → push
stevenson status                # Show store + Redis stats
stevenson ingest file.json --source craigslist  # Import existing data
```

## Scoring Pipeline

The scorer processes paintings in chunks (default 200):
1. Load all stores → collect unscored paintings (skip if `art_score` exists)
2. Load CLIP ViT-L/14 + LAION aesthetic MLP once
3. Per chunk: download images (parallel, 8 threads) → CLIP encode → score → push to Redis → save to store
4. **Art score formula**: `LAION_calibrated * 0.5 + quality * 0.3 + uniqueness * 0.2`
5. LAION calibration: fixed mapping [3, 8] → [0, 100] (not relative min/max)
6. Uniqueness: within-chunk N×N similarity matrix

## Key Patterns

- **Terminal aesthetic** — IBM Plex Mono, dark theme on analytics/data pages
- **No build step** — Static HTML + Vercel serverless TS functions
- **Store = source of truth** — `data/*.json` stores accumulate listings and scores
- **paintings.json is an export** — Generated from stores, not manually edited
- **Live scoring** — Scorer pushes to Redis per-chunk; API uses ZCOUNT for live counts
- **No tests** — No test suite currently

## Dev

```
vercel dev                     # Run frontend + API locally
python scorer.py --limit 50   # Test scorer on 50 paintings
python push.py                # Push paintings.json to Redis
```

## Environment

- `REDIS_URL` — Redis connection string (also checks `stevenson_REDIS_URL`, `.env.local`)
- Python deps: `torch`, `open_clip_torch`, `pillow`, `numpy`, `requests`, `redis`
- Node deps: `ioredis`, `@vercel/node` (TypeScript)
