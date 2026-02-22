"""
Streaming CLIP-based painting scorer with direct Redis push.

Loads unscored paintings from data stores, scores them in chunks,
pushes each chunk to Redis immediately, and writes scores back to stores.

Usage:
    python scorer.py                        # Score unscored, push each chunk
    python scorer.py --rescore              # Re-score everything
    python scorer.py --chunk-size 500       # Bigger chunks
    python scorer.py --no-push              # Score but don't push
    python scorer.py --limit 100            # Score only first 100 unscored
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

# Lazy imports for CLIP
open_clip = None

DATA_DIR = Path("data")
THUMB_DIR = Path("thumbs")

# LAION calibration: raw scores typically range ~2-9
# Map [3, 8] → [0, 100] with clamp for stable cross-chunk scoring
LAION_CAL_LOW = 3.0
LAION_CAL_HIGH = 8.0


def load_clip():
    """Load CLIP model (ViT-L/14) and preprocessing."""
    global open_clip
    import open_clip as oc
    open_clip = oc

    print("Loading CLIP model (ViT-L/14)...")
    model, _, preprocess = oc.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    tokenizer = oc.get_tokenizer("ViT-L-14")
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)
    print(f"  Model loaded on {device}")

    return model, preprocess, tokenizer, device


# ─── LAION AESTHETIC MLP ───

LAION_WEIGHTS_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
LAION_WEIGHTS_PATH = Path("sac+logos+ava1-l14-linearMSE.pth")


class LAIONAestheticMLP(torch.nn.Module):
    """LAION improved-aesthetic-predictor MLP (768 → 1)."""
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


def download_laion_weights():
    """Download LAION aesthetic predictor weights if not present."""
    if LAION_WEIGHTS_PATH.exists():
        return
    print("  Downloading LAION aesthetic weights...")
    resp = requests.get(LAION_WEIGHTS_URL, timeout=60)
    resp.raise_for_status()
    LAION_WEIGHTS_PATH.write_bytes(resp.content)
    print(f"  Saved {LAION_WEIGHTS_PATH} ({len(resp.content) / 1024:.0f} KB)")


def load_aesthetic_model(device):
    """Load LAION aesthetic predictor."""
    download_laion_weights()
    model = LAIONAestheticMLP(input_size=768).to(device)
    state_dict = torch.load(LAION_WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Loaded LAION aesthetic predictor")
    return model


# ─── CLIP PROMPT-BASED QUALITY SCORING ───

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
    """Score images against quality prompts using CLIP."""
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


# ─── STYLE CLASSIFICATION VIA CLIP ───

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
    """Classify painting style using CLIP zero-shot."""
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


# ─── ARTIST NAME PARSING ───

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
    """Extract artist name from painting title."""
    if not title:
        return None, None

    t = title.strip()

    m = re.search(r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})', t)
    if m:
        name = _clean_name(m.group(1).strip())
        first_word = name.split()[0].lower() if name else ""
        if first_word not in BY_EXCLUSIONS and _valid_name(name):
            return name, "high"

    m = re.search(
        r'\b[Ss][Ii][Gg][Nn][Ee][Dd]\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})',
        t,
    )
    if not m:
        m = re.search(
            r'\b[Ss][Ii][Gg][Nn][Aa][Tt][Uu][Rr][Ee]\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})',
            t,
        )
    if m:
        name = _clean_name(m.group(1).strip())
        if _valid_name(name):
            return name, "medium"

    m = re.search(
        r'\b[Aa][Rr][Tt][Ii][Ss][Tt]:?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})',
        t,
    )
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


# ─── VALUE SCORE ───

def compute_value_score(art_score, price):
    if art_score is None or price is None or price <= 0:
        return None
    return round((art_score / 100) / math.log(price + 1) * 100, 1)


# ─── IMAGE DOWNLOAD (parallel) ───

def _download_one(painting, dest_path):
    """Download a single painting thumbnail. Returns True on success."""
    url = painting.get("images", [None])[0]
    if not url:
        return False
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def download_chunk_images(paintings, chunk_dir):
    """Download images for a chunk in parallel. Returns {painting_id: image_path}."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for p in paintings:
            pid = p.get("id", "")
            if not pid or not p.get("images"):
                continue
            # Use sanitized filename
            safe_name = pid.replace(":", "_") + ".jpg"
            img_path = chunk_dir / safe_name
            if img_path.exists():
                results[pid] = img_path
                continue
            fut = pool.submit(_download_one, p, img_path)
            futures[fut] = (pid, img_path)

        for fut in as_completed(futures):
            pid, img_path = futures[fut]
            if fut.result():
                results[pid] = img_path

    return results


# ─── STORE I/O ───

def load_all_stores():
    """Load all data stores, return {source: store_dict}."""
    DATA_DIR.mkdir(exist_ok=True)
    stores = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        source = path.stem
        stores[source] = json.loads(path.read_text())
    return stores


def save_store(source, store):
    """Save a store back to disk."""
    path = DATA_DIR / f"{source}.json"
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False))


def collect_unscored(stores, rescore=False):
    """Collect paintings that need scoring from all stores.

    Returns list of (source, painting_id, painting_dict) tuples.
    """
    items = []
    for source, store in stores.items():
        for pid, p in store["listings"].items():
            if not p.get("images"):
                continue
            if not rescore and p.get("art_score") is not None:
                continue
            items.append((source, pid, p))
    return items


# ─── CHUNK SCORING PIPELINE ───

@torch.no_grad()
def score_chunk(paintings, clip_model, preprocess, tokenizer, aesthetic_model, device, chunk_dir):
    """Score a chunk of paintings. Returns list of enriched painting dicts."""
    # Download images
    image_map = download_chunk_images(paintings, chunk_dir)

    # Encode images with CLIP
    all_features = []
    valid_indices = []  # indices into paintings list
    batch_images = []
    batch_indices = []
    batch_size = 16

    for i, p in enumerate(paintings):
        pid = p.get("id", "")
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

    # Save CLIP embeddings as bytes for vector search
    embeddings = {}
    feat_np = image_features.cpu().numpy()
    for j, idx in enumerate(valid_indices):
        pid = paintings[idx].get("id", "")
        if pid:
            embeddings[pid] = feat_np[j].astype(np.float32).tobytes()

    # LAION aesthetic scoring with fixed calibration
    aesthetic_raw = aesthetic_model(image_features).squeeze(-1).cpu().numpy()
    laion_cal = np.clip(
        (aesthetic_raw - LAION_CAL_LOW) / (LAION_CAL_HIGH - LAION_CAL_LOW) * 100,
        0, 100,
    )

    # Prompt quality scoring
    prompt_scores = compute_prompt_scores(clip_model, tokenizer, image_features, device)

    # Style classification
    styles = classify_styles(clip_model, tokenizer, image_features, device)

    # Within-chunk uniqueness (N×N similarity)
    sim_matrix = (image_features @ image_features.T).cpu().numpy()
    np.fill_diagonal(sim_matrix, 0)
    avg_similarity = sim_matrix.mean(axis=1)
    uniqueness = ((1 - avg_similarity) * 100).clip(0, 100)

    # Enrich paintings
    scored = []
    for j, idx in enumerate(valid_indices):
        p = paintings[idx]

        # Parse artist
        artist, confidence = parse_artist(p.get("title", ""))
        p["artist"] = artist
        p["artist_confidence"] = confidence

        # Scores
        p["aesthetic_score"] = round(float(aesthetic_raw[j]), 2)
        p["quality_score"] = round(float(prompt_scores[j]), 1)
        p["clip_styles"] = styles[j]
        p["uniqueness"] = round(float(uniqueness[j]), 1)

        # Composite: LAION 50% + quality 30% + uniqueness 20%
        ln = laion_cal[j]
        pq = prompt_scores[j]
        u = uniqueness[j]
        p["art_score"] = round(float(ln * 0.5 + pq * 0.3 + u * 0.2), 1)

        # Value score
        p["value_score"] = compute_value_score(p["art_score"], p.get("price"))

        scored.append(p)

    return scored, embeddings


# ─── STREAMING SCORER ───

class StreamingScorer:
    """Holds pre-loaded models + Redis and scores paintings as they arrive.

    Used by `stevenson update --stream` to score scraper batches incrementally
    instead of waiting for all scraping to finish.
    """

    def __init__(self, chunk_size=200):
        print("\nLoading models...")
        self.clip_model, self.preprocess, self.tokenizer, self.device = load_clip()
        self.aesthetic_model = load_aesthetic_model(self.device)

        # Connect to Redis
        self.redis_client = None
        self.redis_binary = None
        try:
            import redis
            from push import get_redis_url, get_redis_binary, ensure_vector_index
            self.redis_client = redis.from_url(get_redis_url(), decode_responses=True)
            self.redis_client.ping()
            print("Connected to Redis")
            self.redis_binary = get_redis_binary()
            ensure_vector_index(self.redis_binary)
        except Exception as e:
            print(f"Warning: Redis unavailable ({e}), scoring without push")

        self.chunk_size = chunk_size
        self.buffer = []  # (source_name, pid, painting_dict) tuples
        self.stores = {}  # source_name -> store_dict, for writeback
        self.chunk_idx = 0
        self.total_scored = 0
        self.total_pushed = 0
        THUMB_DIR.mkdir(exist_ok=True)

    def register_store(self, source, store):
        """Register a store so scored paintings can be saved back."""
        self.stores[source] = store

    def add(self, source, items):
        """Buffer new unscored paintings. Auto-scores when buffer fills.

        items: list of (pid, painting_dict) tuples
        """
        for pid, p in items:
            if p.get("art_score") is None and p.get("images"):
                self.buffer.append((source, pid, p))
        while len(self.buffer) >= self.chunk_size:
            self._flush_chunk(self.buffer[:self.chunk_size])
            self.buffer = self.buffer[self.chunk_size:]

    def flush(self):
        """Score any remaining paintings in the buffer."""
        if self.buffer:
            self._flush_chunk(self.buffer)
            self.buffer = []

    def _flush_chunk(self, items):
        """Score a chunk and push to Redis. Reuses existing score_chunk()."""
        paintings = [p for _, _, p in items]
        self.chunk_idx += 1
        chunk_dir = THUMB_DIR / f"stream_{self.chunk_idx}"

        print(f"\n--- Chunk {self.chunk_idx} ({len(paintings)} paintings) ---")

        scored, embeddings = score_chunk(
            paintings, self.clip_model, self.preprocess, self.tokenizer,
            self.aesthetic_model, self.device, chunk_dir,
        )

        if not scored:
            print("  No images scored in this chunk")
            return

        self.total_scored += len(scored)

        # Push to Redis
        if self.redis_client:
            from push import push_paintings, push_embeddings
            pushed, _ = push_paintings(scored, self.redis_client)
            self.total_pushed += pushed
            emb_count = 0
            if self.redis_binary and embeddings:
                emb_count = push_embeddings(embeddings, self.redis_binary)
            print(f"  Scored {len(scored)} — pushed {pushed} to Redis, "
                  f"{emb_count} embeddings ({self.total_scored} total scored)")
        else:
            print(f"  Scored {len(scored)} ({self.total_scored} total scored)")

        # Save affected stores (score_chunk modifies paintings in-place)
        sources_in_chunk = {source for source, _, _ in items}
        for source in sources_in_chunk:
            if source in self.stores:
                save_store(source, self.stores[source])

    def finalize(self):
        """Push final stats to Redis and print summary."""
        if self.redis_client and self.total_pushed > 0:
            from push import push_stats
            # Load ALL stores (not just registered ones) for accurate stats
            all_stores = load_all_stores()
            all_paintings = []
            for store in all_stores.values():
                all_paintings.extend(
                    p for p in store["listings"].values() if p.get("images")
                )
            stats = push_stats(all_paintings, self.redis_client)
            print(f"\nRedis stats updated: {stats.get('scored_count', 0)} scored, "
                  f"{stats.get('total_listings', 0)} total")

        if self.total_scored:
            print(f"\n{'='*50}")
            print(f"  Streaming scoring complete")
            print(f"{'='*50}")
            print(f"  Total scored: {self.total_scored}")
            if self.redis_client:
                print(f"  Total pushed: {self.total_pushed}")


# ─── MAIN PIPELINE ───

def score_paintings(
    limit=None,
    rescore=False,
    chunk_size=200,
    no_push=False,
):
    """Stream-score paintings from stores, pushing chunks to Redis."""
    # Load stores
    stores = load_all_stores()
    if not stores:
        print("No data stores found in data/")
        return

    total_in_stores = sum(len(s["listings"]) for s in stores.values())
    print(f"Stores: {len(stores)} sources, {total_in_stores:,} total listings")

    # Collect unscored
    items = collect_unscored(stores, rescore=rescore)
    if not items:
        print("All paintings already scored (use --rescore to re-score)")
        return

    if limit:
        items = items[:limit]

    print(f"To score: {len(items):,} paintings (chunk size: {chunk_size})")

    # Load models once
    print("\nLoading models...")
    clip_model, preprocess, tokenizer, device = load_clip()
    aesthetic_model = load_aesthetic_model(device)

    # Connect to Redis (unless --no-push)
    redis_client = None
    redis_binary = None
    if not no_push:
        try:
            import redis
            from push import get_redis_url, get_redis_binary, ensure_vector_index
            redis_client = redis.from_url(get_redis_url(), decode_responses=True)
            redis_client.ping()
            print("Connected to Redis")
            # Binary client for embedding push
            redis_binary = get_redis_binary()
            ensure_vector_index(redis_binary)
        except Exception as e:
            print(f"Warning: Redis unavailable ({e}), scoring without push")
            redis_client = None
            redis_binary = None

    # Process in chunks
    THUMB_DIR.mkdir(exist_ok=True)
    total_scored = 0
    total_pushed = 0
    all_scored_paintings = []
    num_chunks = math.ceil(len(items) / chunk_size)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(items))
        chunk_items = items[start:end]
        chunk_paintings = [p for _, _, p in chunk_items]

        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_paintings)} paintings) ---")

        # Score the chunk
        chunk_dir = THUMB_DIR / f"chunk_{chunk_idx}"
        scored, embeddings = score_chunk(
            chunk_paintings, clip_model, preprocess, tokenizer,
            aesthetic_model, device, chunk_dir,
        )

        if not scored:
            print(f"  No images scored in this chunk")
            continue

        total_scored += len(scored)
        all_scored_paintings.extend(scored)

        # Push to Redis
        if redis_client:
            from push import push_paintings, push_embeddings
            pushed, skipped = push_paintings(scored, redis_client)
            total_pushed += pushed
            emb_count = 0
            if redis_binary and embeddings:
                emb_count = push_embeddings(embeddings, redis_binary)
            print(f"  Scored {len(scored)} — pushed {pushed} to Redis, "
                  f"{emb_count} embeddings ({total_scored}/{len(items)} total)")
        else:
            print(f"  Scored {len(scored)} ({total_scored}/{len(items)} total)")

        # Write scores back to stores
        for source, pid, p in chunk_items:
            if p.get("art_score") is not None:
                stores[source]["listings"][pid] = p

        # Save stores after each chunk (crash-safe)
        sources_in_chunk = {source for source, _, _ in chunk_items}
        for source in sources_in_chunk:
            save_store(source, stores[source])

    # Update Redis stats
    if redis_client and total_pushed > 0:
        from push import push_stats
        # Gather all paintings from stores for accurate stats
        all_paintings = []
        for store in stores.values():
            all_paintings.extend(
                p for p in store["listings"].values() if p.get("images")
            )
        stats = push_stats(all_paintings, redis_client)
        print(f"\nRedis stats updated: {stats.get('scored_count', 0)} scored, "
              f"{stats.get('total_listings', 0)} total")

    # Summary
    if all_scored_paintings:
        scores = [p["art_score"] for p in all_scored_paintings]
        print(f"\n{'='*50}")
        print(f"  Scoring complete")
        print(f"{'='*50}")
        print(f"  Scored: {total_scored}")
        print(f"  Art score range: {min(scores):.1f} – {max(scores):.1f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.1f}")
        if redis_client:
            print(f"  Pushed to Redis: {total_pushed}")

        artists = [p for p in all_scored_paintings if p.get("artist")]
        print(f"  Artists found: {len(artists)}")

        top = sorted(all_scored_paintings, key=lambda p: p["art_score"], reverse=True)[:5]
        print(f"\n  Top 5:")
        for p in top:
            print(f"    {p['art_score']:.1f} | ${p.get('price', '?')} | {p['title'][:55]}")


@torch.no_grad()
def embed_chunk(paintings, clip_model, preprocess, device, chunk_dir):
    """Compute CLIP embeddings only (no scoring). Returns {painting_id: bytes}."""
    image_map = download_chunk_images(paintings, chunk_dir)

    all_features = []
    valid_pids = []
    batch_images = []
    batch_pids = []
    batch_size = 16

    for p in paintings:
        pid = p.get("id", "")
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


def embed_paintings(chunk_size=200):
    """Backfill CLIP embeddings for already-scored paintings."""
    stores = load_all_stores()
    if not stores:
        print("No data stores found")
        return

    # Collect scored paintings
    items = []
    for source, store in stores.items():
        for pid, p in store["listings"].items():
            if not p.get("images") or p.get("art_score") is None:
                continue
            items.append((source, pid, p))

    if not items:
        print("No scored paintings to embed")
        return

    print(f"Found {len(items)} scored paintings")

    # Check which already have embeddings in Redis
    from push import get_redis_binary, ensure_vector_index, push_embeddings
    r_bin = get_redis_binary()

    print("Checking for existing embeddings...")
    pipe = r_bin.pipeline()
    for _, pid, _ in items:
        pipe.hexists(f"stv:p:{pid}", "embedding")
    results = pipe.execute()

    need_embedding = [(s, pid, p) for (s, pid, p), has in zip(items, results) if not has]
    print(f"  {len(need_embedding)} need embeddings "
          f"({len(items) - len(need_embedding)} already have)")

    if not need_embedding:
        print("All embeddings up to date")
        return

    ensure_vector_index(r_bin)

    # Load CLIP model only (no aesthetic model needed)
    print("\nLoading CLIP model...")
    clip_model, preprocess, _, device = load_clip()

    THUMB_DIR.mkdir(exist_ok=True)
    total_pushed = 0
    num_chunks = math.ceil(len(need_embedding) / chunk_size)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(need_embedding))
        chunk_items = need_embedding[start:end]
        chunk_paintings = [p for _, _, p in chunk_items]

        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} "
              f"({len(chunk_paintings)} paintings) ---")

        chunk_dir = THUMB_DIR / f"emb_{chunk_idx}"
        embeddings = embed_chunk(
            chunk_paintings, clip_model, preprocess, device, chunk_dir,
        )

        if embeddings:
            count = push_embeddings(embeddings, r_bin)
            total_pushed += count
            print(f"  Pushed {count} embeddings "
                  f"({total_pushed}/{len(need_embedding)} total)")

    print(f"\nDone! Pushed {total_pushed} embeddings to Redis")


def main():
    parser = argparse.ArgumentParser(description="Streaming CLIP painting scorer")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only first N unscored paintings")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score all paintings (ignore existing scores)")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Paintings per chunk (default: 200)")
    parser.add_argument("--no-push", action="store_true",
                        help="Score without pushing to Redis")
    parser.add_argument("--embed-only", action="store_true",
                        help="Backfill CLIP embeddings for already-scored paintings")
    args = parser.parse_args()

    if args.embed_only:
        embed_paintings(chunk_size=args.chunk_size)
    else:
        score_paintings(
            limit=args.limit,
            rescore=args.rescore,
            chunk_size=args.chunk_size,
            no_push=args.no_push,
        )


if __name__ == "__main__":
    main()
