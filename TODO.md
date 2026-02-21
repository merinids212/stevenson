# TODO — Stevenson

## Scoring (resume on powerful machine)

Scorer was stopped mid-run. ~12,150 paintings still need scoring.

**Current state:**
- 24,098 total listings (9,525 CL + 14,573 eBay)
- ~11,947 scored (9,947 from first full run + 2,000 from second run)
- ~12,150 unscored (mostly new eBay pulls)
- Scores + embeddings written back to stores after each chunk

**To resume:**
```bash
python scorer.py --chunk-size 200
# Will auto-skip already-scored, pick up where it left off
# Pushes each chunk to Redis + saves to stores
# On GPU machine, try --chunk-size 500 for faster throughput
```

## Embeddings backfill

The first 9,947 paintings were scored before embeddings were added.
They need embedding backfill for the FOR YOU recommendation engine.

```bash
python scorer.py --embed-only --chunk-size 200
# Only computes CLIP embeddings, skips paintings that already have them
```

## Data freshness

- Last CL pull: 2026-02-21 (414 regions, all US)
- Last eBay pull: 2026-02-21 (8 queries × 8 pages = 13,460 new)
- Consider running `stevenson pull` daily or setting up a cron

## Features in progress

- **FOR YOU** — like-based recommendations using CLIP KNN vector search
  - `api/like.ts`, `api/recommend.ts` already built
  - Needs embeddings backfill to work (see above)
- **Data page** — live pipeline dashboard at `/data`
  - Shows scoring progress, embedding progress, per-source breakdown
  - Polls `/api/pipeline` every 10s

## Cleanup

- [ ] Remove `paintings.json` from git tracking (stores are source of truth now)
- [ ] Consider adding `thumbs/` to `.gitignore` (already in `.vercelignore`)
