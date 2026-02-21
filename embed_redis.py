"""
Embed paintings directly from Redis — no local data stores needed.

Reads painting image URLs from Redis, downloads images, computes CLIP
embeddings on M4 Max (MPS), and pushes embeddings back to Redis.

Usage:
    python3 embed_redis.py                  # Embed all unembedded paintings
    python3 embed_redis.py --limit 100      # Embed first 100 only
    python3 embed_redis.py --chunk-size 100 # Smaller chunks
"""

import argparse
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

THUMB_DIR = Path("thumbs")


def get_redis_url():
    url = os.environ.get("REDIS_URL") or os.environ.get("stevenson_REDIS_URL")
    if url:
        return url
    env_path = Path(__file__).parent / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("REDIS_URL=") or line.startswith("stevenson_REDIS_URL="):
                return line.split("=", 1)[1].strip().strip('"')
    print("Error: REDIS_URL not set")
    sys.exit(1)


def load_clip():
    import open_clip
    print("Loading CLIP model (ViT-L/14)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)
    print(f"  Model loaded on {device}")
    return model, preprocess, device


def download_one(url, dest_path):
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def download_images(paintings, chunk_dir):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for p in paintings:
            pid = p["id"]
            url = p.get("image_url")
            if not url:
                continue
            safe_name = pid.replace(":", "_") + ".jpg"
            img_path = chunk_dir / safe_name
            if img_path.exists():
                results[pid] = img_path
                continue
            fut = pool.submit(download_one, url, img_path)
            futures[fut] = (pid, img_path)

        for fut in as_completed(futures):
            pid, img_path = futures[fut]
            if fut.result():
                results[pid] = img_path

    return results


@torch.no_grad()
def embed_chunk(paintings, clip_model, preprocess, device, chunk_dir):
    image_map = download_images(paintings, chunk_dir)

    all_features = []
    valid_pids = []
    batch_images = []
    batch_pids = []
    batch_size = 16

    for p in paintings:
        pid = p["id"]
        if pid not in image_map:
            continue
        try:
            img = Image.open(image_map[pid]).convert("RGB")
            tensor = preprocess(img)
            batch_images.append(tensor)
            batch_pids.append(pid)
        except Exception:
            continue

        if len(batch_images) >= batch_size:
            batch_tensor = torch.stack(batch_images).to(device)
            feats = clip_model.encode_image(batch_tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_features.append(feats)
            valid_pids.extend(batch_pids)
            batch_images = []
            batch_pids = []

    if batch_images:
        batch_tensor = torch.stack(batch_images).to(device)
        feats = clip_model.encode_image(batch_tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
        all_features.append(feats)
        valid_pids.extend(batch_pids)

    if not all_features:
        return {}

    all_feats = torch.cat(all_features, dim=0).cpu().numpy()
    return {
        pid: all_feats[j].astype(np.float32).tobytes()
        for j, pid in enumerate(valid_pids)
    }


def main():
    parser = argparse.ArgumentParser(description="Embed paintings from Redis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=200)
    args = parser.parse_args()

    import redis
    from push import ensure_vector_index

    url = get_redis_url()
    r = redis.from_url(url, decode_responses=True)
    r_bin = redis.from_url(url, decode_responses=False)
    r.ping()
    print(f"Connected to Redis")

    # Get all painting IDs
    all_ids = r.zrange("stv:idx:art_score", 0, -1)
    print(f"Total paintings in Redis: {len(all_ids)}")

    # Check which already have embeddings
    print("Checking existing embeddings...")
    pipe = r_bin.pipeline()
    for pid in all_ids:
        pipe.hexists(f"stv:p:{pid}", "embedding")
    results = pipe.execute()

    need_ids = [pid for pid, has in zip(all_ids, results) if not has]
    print(f"  {len(need_ids)} need embeddings ({len(all_ids) - len(need_ids)} already have)")

    if not need_ids:
        print("All embeddings up to date!")
        return

    if args.limit:
        need_ids = need_ids[:args.limit]
        print(f"  Limited to {len(need_ids)}")

    # Fetch image URLs from Redis for paintings that need embedding
    print("Fetching image URLs...")
    pipe = r.pipeline()
    for pid in need_ids:
        pipe.hget(f"stv:p:{pid}", "images")
    img_results = pipe.execute()

    paintings = []
    for pid, images_json in zip(need_ids, img_results):
        if not images_json:
            continue
        try:
            import json
            images = json.loads(images_json)
            if images:
                paintings.append({"id": pid, "image_url": images[0]})
        except Exception:
            continue

    print(f"  {len(paintings)} paintings with image URLs")

    # Ensure vector index
    ensure_vector_index(r_bin)

    # Load CLIP
    clip_model, preprocess, device = load_clip()

    # Process in chunks
    THUMB_DIR.mkdir(exist_ok=True)
    total_pushed = 0
    num_chunks = math.ceil(len(paintings) / args.chunk_size)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, len(paintings))
        chunk = paintings[start:end]

        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ({len(chunk)} paintings) ---")

        chunk_dir = THUMB_DIR / f"emb_{chunk_idx}"
        embeddings = embed_chunk(chunk, clip_model, preprocess, device, chunk_dir)

        if embeddings:
            pipe = r_bin.pipeline()
            for pid, emb_bytes in embeddings.items():
                pipe.hset(f"stv:p:{pid}", "embedding", emb_bytes)
                pipe.sadd("stv:embedded", pid)
            pipe.execute()
            total_pushed += len(embeddings)
            print(f"  Pushed {len(embeddings)} embeddings ({total_pushed}/{len(paintings)} total)")

    print(f"\nDone! Pushed {total_pushed} embeddings to Redis")


if __name__ == "__main__":
    main()
