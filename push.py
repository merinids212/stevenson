"""
Push paintings.json data to Redis Cloud.

Reads the local paintings.json file and pushes all entries to Redis
with proper indexing for the API to query.

Usage:
    python push.py
    python push.py --file paintings.json --flush
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import redis


def get_redis_url() -> str:
    url = os.environ.get("REDIS_URL") or os.environ.get("stevenson_REDIS_URL")
    if url:
        return url
    # Try .env.local
    env_path = Path(__file__).parent / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("REDIS_URL=") or line.startswith("stevenson_REDIS_URL="):
                return line.split("=", 1)[1].strip().strip('"')
    print("Error: REDIS_URL not set")
    sys.exit(1)


def extract_id(url: str) -> str | None:
    """Extract Craigslist post ID from URL."""
    m = re.search(r"/(\d+)\.html", url)
    return m.group(1) if m else None


def push_paintings(paintings: list[dict], r):
    """Push a batch of paintings to Redis (upsert). Returns (pushed, skipped) counts."""
    pipe = r.pipeline()
    pushed = 0
    skipped = 0

    for p in paintings:
        pid = p.get("id") or ""
        if not pid:
            ext_id = extract_id(p.get("url", ""))
            if not ext_id:
                skipped += 1
                continue
            pid = f"cl:{ext_id}"

        source = p.get("source", "cl")
        ext_id = pid.split(":", 1)[1] if ":" in pid else pid

        fields = {
            "title": p.get("title", ""),
            "price": str(p["price"]) if p.get("price") is not None else "",
            "url": p.get("url", ""),
            "location": p.get("location", ""),
            "latitude": str(p["latitude"]) if p.get("latitude") is not None else "",
            "longitude": str(p["longitude"]) if p.get("longitude") is not None else "",
            "images": json.dumps(p.get("images", [])),
            "posted": p.get("posted", ""),
            "region": p.get("region", ""),
            "state": p.get("state", ""),
            "quality_score": str(p["quality_score"]) if p.get("quality_score") is not None else "",
            "clip_styles": json.dumps(p["clip_styles"]) if p.get("clip_styles") is not None else "",
            "uniqueness": str(p["uniqueness"]) if p.get("uniqueness") is not None else "",
            "art_score": str(p["art_score"]) if p.get("art_score") is not None else "",
            "artist": p.get("artist") or "",
            "artist_confidence": str(p["artist_confidence"]) if p.get("artist_confidence") is not None else "",
            "value_score": str(p["value_score"]) if p.get("value_score") is not None else "",
            "aesthetic_score": str(p["aesthetic_score"]) if p.get("aesthetic_score") is not None else "",
            "source": source,
            "external_id": ext_id,
        }

        pipe.hset(f"stv:p:{pid}", mapping=fields)
        pipe.sadd("stv:dedup", pid)

        art_score = p.get("art_score") or 0
        pipe.zadd("stv:idx:art_score", {pid: art_score})

        price = p.get("price") or 0
        pipe.zadd("stv:idx:price", {pid: price})

        if p.get("value_score") is not None:
            pipe.zadd("stv:idx:value_score", {pid: p["value_score"]})

        region = p.get("region", "")
        if region:
            pipe.zadd(f"stv:idx:region:{region}", {pid: art_score})

        state = p.get("state", "")
        if state:
            pipe.zadd(f"stv:idx:state:{state}", {pid: art_score})

        pipe.zadd(f"stv:idx:source:{source}", {pid: art_score})
        pushed += 1

    pipe.execute()
    return pushed, skipped


def _median(vals):
    """Compute true median (average of two middle values for even-length lists)."""
    if not vals:
        return 0
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2


def push_stats(paintings: list[dict], r):
    """Recompute and push stv:stats from a list of paintings."""
    prices = sorted(p["price"] for p in paintings if p.get("price") is not None and p["price"] > 0)
    scored = [p for p in paintings if p.get("art_score") is not None]
    art_scores = sorted(p["art_score"] for p in scored)
    regions = {p.get("region", "") for p in paintings} - {""}
    states = {p.get("state", "") for p in paintings} - {""}
    artists = [p for p in paintings if p.get("artist")]
    top_rated = [p for p in scored if p["art_score"] >= 55]
    gems = [p for p in paintings if p.get("value_score") is not None and p["value_score"] >= 8]

    stats = {
        "total_listings": str(len(paintings)),
        "regions": str(len(regions)),
        "states": str(len(states)),
        "price_min": str(min(prices)) if prices else "0",
        "price_max": str(max(prices)) if prices else "0",
        "price_median": str(_median(prices)),
        "scored_count": str(len(scored)),
        "artists_count": str(len(artists)),
        "top_rated_count": str(len(top_rated)),
        "gems_count": str(len(gems)),
        "median_art_score": str(_median(art_scores)),
    }

    r.hset("stv:stats", mapping=stats)
    return stats


def push(file_path: str = "paintings.json", flush: bool = False):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    data = json.loads(path.read_text())
    print(f"Loaded {len(data)} paintings from {path}")

    r = redis.from_url(get_redis_url(), decode_responses=True)
    r.ping()
    print("Connected to Redis")

    if flush:
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = r.scan(cursor, match="stv:*", count=500)
            if keys:
                r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        print(f"Flushed {deleted} existing keys")

    paintings = [p for p in data if p.get("images") and len(p["images"]) > 0]
    print(f"Pushing {len(paintings)} paintings with images...")

    pushed, skipped = push_paintings(paintings, r)
    print(f"Pushed {pushed} paintings ({skipped} skipped, no ID)")

    print("Computing stats...")
    stats = push_stats(paintings, r)
    print(f"Stats: {json.dumps(stats, indent=2)}")
    print("Done!")


def get_redis_binary():
    """Get a Redis client for binary operations (no decode_responses)."""
    return redis.from_url(get_redis_url(), decode_responses=False)


def ensure_vector_index(r_bin):
    """Create RediSearch vector index for CLIP embeddings if it doesn't exist."""
    try:
        r_bin.execute_command(
            "FT.CREATE", "stv:vec_idx",
            "ON", "HASH",
            "PREFIX", "1", "stv:p:",
            "SCHEMA",
            "embedding", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", "768", "DISTANCE_METRIC", "COSINE",
        )
        print("  Created vector search index stv:vec_idx")
    except redis.ResponseError as e:
        if "Index already exists" in str(e):
            pass
        else:
            raise


def push_embeddings(embeddings: dict, r_bin) -> int:
    """Push CLIP embeddings to Redis hashes. embeddings = {painting_id: float32_bytes}."""
    if not embeddings:
        return 0
    pipe = r_bin.pipeline()
    for pid, emb_bytes in embeddings.items():
        pipe.hset(f"stv:p:{pid}", "embedding", emb_bytes)
        pipe.sadd("stv:embedded", pid)  # track which paintings have embeddings
    pipe.execute()
    return len(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push paintings to Redis")
    parser.add_argument("--file", default="paintings.json", help="Path to paintings.json")
    parser.add_argument("--flush", action="store_true", help="Delete all stv: keys before pushing")
    args = parser.parse_args()

    push(args.file, args.flush)
