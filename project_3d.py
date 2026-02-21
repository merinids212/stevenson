#!/usr/bin/env python3
"""
Compute UMAP 3D projections of painting CLIP embeddings from Redis.
Stores x, y, z coordinates back into each painting's Redis hash.
Also writes a compact JSON file for the taste visualization page.

Usage:
    python project_3d.py                # Project all embedded paintings
    python project_3d.py --limit 5000   # Project a subset for testing
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import redis
import umap


def get_redis():
    url = (
        os.environ.get("REDIS_URL")
        or os.environ.get("stevenson_REDIS_URL")
        or ""
    )
    if not url:
        # Try .env.local
        env_path = os.path.join(os.path.dirname(__file__), ".env.local")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("REDIS_URL="):
                        url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("stevenson_REDIS_URL=") and not url:
                        url = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not url:
        print("No REDIS_URL found")
        sys.exit(1)
    return redis.from_url(url, decode_responses=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of paintings")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    args = parser.parse_args()

    r = get_redis()
    print("Connected to Redis")

    # Get all embedded painting IDs
    embedded_ids = r.smembers(b"stv:embedded")
    embedded_ids = [eid.decode() if isinstance(eid, bytes) else eid for eid in embedded_ids]
    print(f"  {len(embedded_ids)} paintings with embeddings")

    if args.limit:
        embedded_ids = embedded_ids[: args.limit]
        print(f"  Limited to {len(embedded_ids)}")

    # Fetch embeddings + metadata in batches
    DIM = 768
    ids = []
    embeddings = []
    metadata = []  # art_score, price, clip_styles, title, region, source, images
    BATCH = 500

    print("Fetching embeddings from Redis...")
    for i in range(0, len(embedded_ids), BATCH):
        batch_ids = embedded_ids[i : i + BATCH]
        pipe = r.pipeline()
        for pid in batch_ids:
            pipe.hget(f"stv:p:{pid}", "embedding")
            pipe.hmget(
                f"stv:p:{pid}",
                "art_score", "price", "clip_styles", "title",
                "region", "source", "images", "url", "artist",
                "value_score",
            )
        results = pipe.execute()

        for j in range(0, len(results), 2):
            emb_data = results[j]
            meta_data = results[j + 1]
            pid = batch_ids[j // 2]

            if emb_data and len(emb_data) >= DIM * 4:
                vec = np.frombuffer(emb_data[:DIM * 4], dtype=np.float32)
                ids.append(pid)
                embeddings.append(vec)

                art_score = float(meta_data[0]) if meta_data[0] else 0
                price = float(meta_data[1]) if meta_data[1] else 0
                styles_raw = meta_data[2].decode() if meta_data[2] else "[]"
                title = meta_data[3].decode() if meta_data[3] else ""
                region = meta_data[4].decode() if meta_data[4] else ""
                source = meta_data[5].decode() if meta_data[5] else ""
                images = meta_data[6].decode() if meta_data[6] else "[]"
                url = meta_data[7].decode() if meta_data[7] else ""
                artist = meta_data[8].decode() if meta_data[8] else ""
                value_score = float(meta_data[9]) if meta_data[9] else 0

                # Parse primary style
                try:
                    styles = json.loads(styles_raw)
                    primary_style = styles[0]["style"] if styles else ""
                except Exception:
                    primary_style = ""

                # Parse first image
                try:
                    img_list = json.loads(images)
                    thumb = img_list[0] if img_list else ""
                except Exception:
                    thumb = ""

                metadata.append({
                    "s": round(art_score, 1),    # art_score
                    "p": round(price),            # price
                    "st": primary_style,          # style
                    "t": title[:60],              # title (truncated)
                    "r": region,                  # region
                    "src": source,                # source
                    "img": thumb,                 # thumbnail
                    "u": url,                     # listing url
                    "a": artist,                  # artist
                    "v": round(value_score, 1),   # value score
                })

        done = min(i + BATCH, len(embedded_ids))
        print(f"  {done}/{len(embedded_ids)} fetched")

    X = np.array(embeddings)
    print(f"\nRunning UMAP on {X.shape[0]} × {X.shape[1]} matrix...")
    print(f"  n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")
    t0 = time.time()

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    coords_3d = reducer.fit_transform(X)
    elapsed = time.time() - t0
    print(f"  UMAP done in {elapsed:.1f}s")

    # Normalize to [-1, 1] range
    mins = coords_3d.min(axis=0)
    maxs = coords_3d.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    coords_3d = 2 * (coords_3d - mins) / ranges - 1

    # Push x, y, z back to Redis
    print("Pushing 3D coordinates to Redis...")
    pipe = r.pipeline()
    for i, pid in enumerate(ids):
        x, y, z = coords_3d[i]
        pipe.hset(f"stv:p:{pid}", mapping={
            "umap_x": str(round(float(x), 4)),
            "umap_y": str(round(float(y), 4)),
            "umap_z": str(round(float(z), 4)),
        })
        if (i + 1) % BATCH == 0:
            pipe.execute()
            pipe = r.pipeline()
            print(f"  {i + 1}/{len(ids)} pushed")
    pipe.execute()
    print(f"  {len(ids)}/{len(ids)} pushed")

    # Write compact JSON for the visualization
    # Format: { ids: [...], coords: [x,y,z,...], meta: [...] }
    output = {
        "ids": ids,
        "coords": [round(float(c), 4) for row in coords_3d for c in row],  # flat [x,y,z,x,y,z,...]
        "meta": metadata,
    }

    out_path = os.path.join(os.path.dirname(__file__), "taste_map.json")
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.1f} MB, {len(ids)} paintings)")
    print("Done!")


if __name__ == "__main__":
    main()
