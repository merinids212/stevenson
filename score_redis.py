"""
Score paintings directly from Redis — no local data stores needed.

Reads unscored paintings from Redis, downloads images, computes CLIP
embeddings + LAION aesthetic scores + quality/style/uniqueness, and
pushes enriched data back to Redis.

Usage:
    python3 score_redis.py                  # Score all unscored paintings
    python3 score_redis.py --limit 100      # Score first 100 only
    python3 score_redis.py --chunk-size 100 # Smaller chunks
    python3 score_redis.py --rescore        # Re-score everything
"""

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

THUMB_DIR = Path("thumbs")

# LAION calibration: raw scores typically range ~2-9
# Map [3, 8] → [0, 100] with clamp for stable cross-chunk scoring
LAION_CAL_LOW = 3.0
LAION_CAL_HIGH = 8.0

# LAION aesthetic MLP weights
LAION_WEIGHTS_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
LAION_WEIGHTS_PATH = Path("sac+logos+ava1-l14-linearMSE.pth")


# ─── REDIS ───

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


# ─── MODELS ───

def load_clip():
    import open_clip
    print("Loading CLIP model (ViT-L/14)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)
    print(f"  Model loaded on {device}")
    return model, preprocess, tokenizer, device


class LAIONAestheticMLP(torch.nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def load_aesthetic_model(device):
    if not LAION_WEIGHTS_PATH.exists():
        print("  Downloading LAION aesthetic weights...")
        resp = requests.get(LAION_WEIGHTS_URL, timeout=60)
        resp.raise_for_status()
        LAION_WEIGHTS_PATH.write_bytes(resp.content)
    model = LAIONAestheticMLP(input_size=768).to(device)
    state_dict = torch.load(LAION_WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Loaded LAION aesthetic predictor")
    return model


# ─── CLIP SCORING ───

QUALITY_PROMPTS = {
    "high": [
        "a beautiful fine art painting in a museum",
        "a masterpiece painting with excellent composition",
        "professional gallery artwork with rich colors",
        "high quality original painting with skilled brushwork",
        "a stunning artwork by a talented artist",
    ],
    "low": [
        "a cheap amateur painting",
        "a low quality blurry photo of random items",
        "a bad painting with poor composition",
        "arts and crafts project by a beginner",
        "a mass produced print from a store",
    ],
}


@torch.no_grad()
def compute_prompt_scores(model, tokenizer, image_features, device):
    high_texts = tokenizer(QUALITY_PROMPTS["high"]).to(device)
    low_texts = tokenizer(QUALITY_PROMPTS["low"]).to(device)
    high_feats = model.encode_text(high_texts)
    low_feats = model.encode_text(low_texts)
    high_feats /= high_feats.norm(dim=-1, keepdim=True)
    low_feats /= low_feats.norm(dim=-1, keepdim=True)
    high_sim = (image_features @ high_feats.T).mean(dim=-1)
    low_sim = (image_features @ low_feats.T).mean(dim=-1)
    raw = high_sim - low_sim
    scores = ((raw + 0.15) / 0.30 * 100).clamp(0, 100)
    return scores.cpu().numpy()


STYLE_PROMPTS = {
    "abstract": "an abstract painting with geometric shapes and bold colors",
    "impressionist": "an impressionist painting with soft brushstrokes and light",
    "modern": "a modern contemporary art piece",
    "realist": "a realistic painting that looks like a photograph",
    "folk": "a folk art or naive painting",
    "portrait": "a portrait painting of a person",
    "landscape": "a landscape painting of nature or scenery",
    "still_life": "a still life painting of objects or flowers",
    "surreal": "a surrealist painting with dreamlike imagery",
    "pop_art": "a pop art piece with bright colors and bold outlines",
}


@torch.no_grad()
def classify_styles(model, tokenizer, image_features, device):
    texts = tokenizer(list(STYLE_PROMPTS.values())).to(device)
    text_feats = model.encode_text(texts)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    sims = image_features @ text_feats.T
    probs = sims.softmax(dim=-1).cpu().numpy()
    style_names = list(STYLE_PROMPTS.keys())
    results = []
    for i in range(len(probs)):
        top_indices = probs[i].argsort()[::-1][:3]
        styles = [
            {"style": style_names[idx], "confidence": round(float(probs[i][idx]), 3)}
            for idx in top_indices
            if probs[i][idx] > 0.05
        ]
        results.append(styles)
    return results


# ─── ARTIST PARSING ───

BY_EXCLUSIONS = {
    "numbers", "hand", "the", "a", "an", "owner", "seller", "appointment",
    "local", "estate", "unknown", "anonymous", "various", "multiple",
    "me", "us", "them", "request", "design", "nature", "mail",
    "cal", "so", "la",
}

KNOWN_ARTISTS = {
    "picasso", "monet", "renoir", "cezanne", "matisse", "kandinsky",
    "warhol", "basquiat", "hockney", "pollock", "rothko", "koons",
    "banksy", "dali", "miro", "chagall", "rembrandt", "vermeer",
    "caravaggio", "klimt", "schiele", "mondrian", "magritte",
    "lichtenstein", "rockwell", "wyeth", "homer", "hopper",
    "okeefe", "o'keeffe", "rivera", "kahlo", "botero",
    "thiebaud", "diebenkorn", "ruscha", "baldessari",
}

NAME_STOP_WORDS = {
    "oil", "on", "canvas", "acrylic", "watercolor", "painting", "print",
    "lithograph", "etching", "engraving", "serigraph", "giclee",
    "original", "framed", "signed", "large", "small", "vintage", "antique",
    "unfinished", "matted", "mounted",
    "art", "artwork", "piece", "gallery", "museum", "rare", "proof",
    "style", "scene", "geometric", "impressionist", "surreal",
    "beautiful", "stunning", "gorgeous", "amazing", "exceptional", "fine",
    "great", "new", "old", "modern", "contemporary",
    "abstract", "landscape", "portrait", "still", "life", "floral",
    "with", "and", "the", "from", "for", "in", "of", "at", "to", "or",
    "mid", "century", "listed", "numbered", "hand", "painted", "wrap",
    "tiger", "cubs", "cat", "cats", "dog", "horse", "flower", "flowers",
    "raccoon", "fish", "clown", "bird", "pirate", "ship", "street",
    "hong", "kong", "paris", "east", "coast", "native", "american",
    "who", "has", "work", "archetype", "best", "offer",
    "jewish", "men", "boy", "blue", "fisherman", "lake", "nostalgic",
    "number",
}

NAME_REJECTS = {
    "artist", "by artist", "the artist", "by the artist",
    "surf", "proof", "signature", "israeli artist",
}

_NAME_PREFIX_STRIP = {
    "artist", "listed", "california", "american", "haitian", "cuban",
    "french", "italian", "chinese", "canadian", "mexican", "spanish",
    "local", "renowned", "famous", "famed",
}


def _clean_name(name):
    words = name.split()
    while words and words[0].lower() in _NAME_PREFIX_STRIP:
        words.pop(0)
    while words and words[-1].lower() in NAME_STOP_WORDS:
        words.pop()
    return " ".join(words)


def _valid_name(name):
    if not name or len(name) < 3:
        return False
    if name.lower() in NAME_REJECTS:
        return False
    words = name.split()
    if len(words) == 1 and len(name) < 4:
        return False
    return True


def parse_artist(title):
    if not title:
        return None, None
    t = title.strip()
    m = re.search(r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})', t)
    if m:
        name = _clean_name(m.group(1).strip())
        first_word = name.split()[0].lower() if name else ""
        if first_word not in BY_EXCLUSIONS and _valid_name(name):
            return name, "high"
    m = re.search(r'\b[Ss][Ii][Gg][Nn][Ee][Dd]\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})', t)
    if not m:
        m = re.search(r'\b[Ss][Ii][Gg][Nn][Aa][Tt][Uu][Rr][Ee]\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})', t)
    if m:
        name = _clean_name(m.group(1).strip())
        if _valid_name(name):
            return name, "medium"
    m = re.search(r'\b[Aa][Rr][Tt][Ii][Ss][Tt]:?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})', t)
    if m:
        name = _clean_name(m.group(1).strip())
        if _valid_name(name):
            return name, "medium"
    t_lower = t.lower()
    for artist in KNOWN_ARTISTS:
        pattern = r'\b' + re.escape(artist) + r'\b'
        match = re.search(pattern, t_lower)
        if match:
            idx = match.start()
            found = t[idx:idx + len(artist)]
            return found.title(), "low"
    return None, None


def compute_value_score(art_score, price):
    if art_score is None or price is None or price <= 0:
        return None
    return round((art_score / 100) / math.log(price + 1) * 100, 1)


# ─── IMAGE DOWNLOAD ───

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


# ─── CHUNK SCORING ───

@torch.no_grad()
def score_chunk(paintings, clip_model, preprocess, tokenizer, aesthetic_model, device, chunk_dir):
    """Score a chunk: download images, CLIP encode, score, return enriched dicts + embeddings."""
    image_map = download_images(paintings, chunk_dir)

    all_features = []
    valid_indices = []
    batch_images = []
    batch_indices = []
    batch_size = 16

    for i, p in enumerate(paintings):
        pid = p["id"]
        if pid not in image_map:
            continue
        try:
            img = Image.open(image_map[pid]).convert("RGB")
            tensor = preprocess(img)
            batch_images.append(tensor)
            batch_indices.append(i)
        except Exception:
            continue

        if len(batch_images) >= batch_size:
            batch_tensor = torch.stack(batch_images).to(device)
            feats = clip_model.encode_image(batch_tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_features.append(feats)
            valid_indices.extend(batch_indices)
            batch_images = []
            batch_indices = []

    if batch_images:
        batch_tensor = torch.stack(batch_images).to(device)
        feats = clip_model.encode_image(batch_tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
        all_features.append(feats)
        valid_indices.extend(batch_indices)

    if not all_features:
        return [], {}

    image_features = torch.cat(all_features, dim=0)

    # Embeddings
    embeddings = {}
    feat_np = image_features.cpu().numpy()
    for j, idx in enumerate(valid_indices):
        pid = paintings[idx]["id"]
        embeddings[pid] = feat_np[j].astype(np.float32).tobytes()

    # LAION aesthetic
    aesthetic_raw = aesthetic_model(image_features).squeeze(-1).cpu().numpy()
    laion_cal = np.clip(
        (aesthetic_raw - LAION_CAL_LOW) / (LAION_CAL_HIGH - LAION_CAL_LOW) * 100,
        0, 100,
    )

    # Quality scoring
    prompt_scores = compute_prompt_scores(clip_model, tokenizer, image_features, device)

    # Style classification
    styles = classify_styles(clip_model, tokenizer, image_features, device)

    # Uniqueness (within-chunk N×N similarity)
    sim_matrix = (image_features @ image_features.T).cpu().numpy()
    np.fill_diagonal(sim_matrix, 0)
    avg_similarity = sim_matrix.mean(axis=1)
    uniqueness = ((1 - avg_similarity) * 100).clip(0, 100)

    # Build scored results
    scored = []
    for j, idx in enumerate(valid_indices):
        p = dict(paintings[idx])  # copy
        artist, confidence = parse_artist(p.get("title", ""))
        p["artist"] = artist
        p["artist_confidence"] = confidence
        p["aesthetic_score"] = round(float(aesthetic_raw[j]), 2)
        p["quality_score"] = round(float(prompt_scores[j]), 1)
        p["clip_styles"] = styles[j]
        p["uniqueness"] = round(float(uniqueness[j]), 1)

        ln = laion_cal[j]
        pq = prompt_scores[j]
        u = uniqueness[j]
        p["art_score"] = round(float(ln * 0.5 + pq * 0.3 + u * 0.2), 1)
        p["value_score"] = compute_value_score(p["art_score"], p.get("price"))

        scored.append(p)

    return scored, embeddings


# ─── MAIN ───

def main():
    parser = argparse.ArgumentParser(description="Score paintings from Redis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--rescore", action="store_true", help="Re-score all paintings")
    args = parser.parse_args()

    import redis as redis_lib
    from push import ensure_vector_index

    url = get_redis_url()
    r = redis_lib.from_url(url, decode_responses=True)
    r_bin = redis_lib.from_url(url, decode_responses=False)
    r.ping()
    print(f"Connected to Redis")

    # Get all painting IDs
    all_ids = r.zrange("stv:idx:art_score", 0, -1)
    print(f"Total paintings in Redis: {len(all_ids)}")

    if args.rescore:
        need_ids = all_ids
        print(f"  Rescoring all {len(need_ids)} paintings")
    else:
        # Find unscored: art_score == 0
        need_ids = r.zrangebyscore("stv:idx:art_score", 0, 0)
        already_scored = len(all_ids) - len(need_ids)
        print(f"  {len(need_ids)} need scoring ({already_scored} already scored)")

    if not need_ids:
        print("All paintings scored!")
        return

    if args.limit:
        need_ids = need_ids[:args.limit]
        print(f"  Limited to {len(need_ids)}")

    # Fetch painting data from Redis
    print("Fetching painting data from Redis...")
    pipe = r.pipeline()
    for pid in need_ids:
        pipe.hgetall(f"stv:p:{pid}")
    raw_results = pipe.execute()

    paintings = []
    for pid, data in zip(need_ids, raw_results):
        if not data:
            continue
        images_str = data.get("images", "[]")
        try:
            images = json.loads(images_str)
        except Exception:
            images = []
        if not images:
            continue
        paintings.append({
            "id": pid,
            "title": data.get("title", ""),
            "price": float(data["price"]) if data.get("price") else None,
            "url": data.get("url", ""),
            "location": data.get("location", ""),
            "region": data.get("region", ""),
            "state": data.get("state", ""),
            "source": data.get("source", ""),
            "image_url": images[0],
            "images": images,
        })

    print(f"  {len(paintings)} paintings with image URLs")

    # Ensure vector index
    ensure_vector_index(r_bin)

    # Load models
    print("\nLoading models...")
    clip_model, preprocess, tokenizer, device = load_clip()
    aesthetic_model = load_aesthetic_model(device)

    # Process in chunks
    THUMB_DIR.mkdir(exist_ok=True)
    total_scored = 0
    total_pushed = 0
    total_embedded = 0
    num_chunks = math.ceil(len(paintings) / args.chunk_size)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, len(paintings))
        chunk = paintings[start:end]

        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ({len(chunk)} paintings) ---")

        chunk_dir = THUMB_DIR / f"score_{chunk_idx}"
        scored, embeddings = score_chunk(
            chunk, clip_model, preprocess, tokenizer,
            aesthetic_model, device, chunk_dir,
        )

        if not scored:
            print(f"  No images scored in this chunk")
            continue

        total_scored += len(scored)

        # Push scores to Redis
        from push import push_paintings, push_embeddings
        pushed, skipped = push_paintings(scored, r)
        total_pushed += pushed

        # Push embeddings
        emb_count = 0
        if embeddings:
            emb_count = push_embeddings(embeddings, r_bin)
            total_embedded += emb_count

        print(f"  Scored {len(scored)} — pushed {pushed} to Redis, "
              f"{emb_count} embeddings ({total_scored}/{len(paintings)} total)")

    # Summary
    if total_scored > 0:
        print(f"\n{'='*50}")
        print(f"  Scoring complete")
        print(f"{'='*50}")
        print(f"  Scored: {total_scored}")
        print(f"  Pushed: {total_pushed}")
        print(f"  Embeddings: {total_embedded}")

    print(f"\nDone! Scored {total_scored} paintings")


if __name__ == "__main__":
    main()
