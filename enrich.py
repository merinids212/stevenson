"""
Post-scoring enrichment pipeline for paintings.

Adds pyiqa aesthetic scores, CLIP zero-shot tags (subjects, moods, mediums),
dominant color extraction, and search vocabulary encoding.

Usage:
    python enrich.py                        # Enrich all scored but unenriched
    python enrich.py --force                # Re-enrich everything
    python enrich.py --limit 50             # Enrich first 50
    python enrich.py --no-push              # Enrich without pushing to Redis
"""

import argparse
import colorsys
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scorer import (
    load_clip,
    load_all_stores,
    save_store,
    download_chunk_images,
    THUMB_DIR,
    DATA_DIR,
)

# ─── PROMPT DICTIONARIES ───

SUBJECT_PROMPTS = {
    "portrait": "a portrait painting of a person or face",
    "landscape": "a landscape painting of nature, mountains, or scenery",
    "still_life": "a still life painting of objects, flowers, or food",
    "animal": "a painting featuring an animal",
    "figure": "a painting of human figures or bodies",
    "seascape": "a painting of the ocean, sea, or coastal scene",
    "cityscape": "a painting of a city, buildings, or urban scene",
    "floral": "a painting of flowers or botanical subjects",
    "abstract": "an abstract painting with shapes, colors, and patterns",
    "nude": "a nude figure painting",
}

MOOD_PROMPTS = {
    "serene": "a calm, peaceful, and serene painting",
    "dramatic": "a dramatic, intense painting with strong contrast",
    "melancholy": "a sad, melancholy painting with muted tones",
    "joyful": "a bright, happy, and joyful painting",
    "mysterious": "a dark, mysterious, enigmatic painting",
    "romantic": "a romantic, dreamy, soft painting",
}

MEDIUM_PROMPTS = {
    "oil": "an oil painting on canvas",
    "watercolor": "a watercolor painting on paper",
    "acrylic": "an acrylic painting",
    "pastel": "a pastel drawing or painting",
    "ink": "an ink drawing or pen illustration",
    "mixed_media": "a mixed media artwork or collage",
}

COLOR_MAP = {
    "red": (345, 15),
    "orange": (15, 45),
    "yellow": (45, 65),
    "green": (65, 170),
    "teal": (170, 200),
    "blue": (200, 260),
    "purple": (260, 310),
    "pink": (310, 345),
}

SEARCH_TERMS = [
    # Subjects
    "portrait", "landscape", "still life", "abstract", "figure", "nude",
    "seascape", "cityscape", "floral", "animal", "botanical", "maritime",
    "interior", "exterior", "genre scene", "mythology", "religious",
    "historical", "allegorical", "self portrait",
    # Styles
    "impressionist", "expressionist", "realist", "surrealist", "cubist",
    "minimalist", "pop art", "art deco", "art nouveau", "baroque",
    "romantic", "classical", "modern", "contemporary", "postmodern",
    "folk art", "naive art", "outsider art", "street art", "photorealism",
    # Mediums
    "oil painting", "watercolor", "acrylic", "pastel", "ink drawing",
    "charcoal", "pencil drawing", "mixed media", "collage", "gouache",
    "tempera", "fresco", "encaustic", "digital art", "print",
    "lithograph", "etching", "woodcut", "screen print", "monotype",
    # Colors
    "red painting", "blue painting", "green painting", "yellow painting",
    "orange painting", "purple painting", "pink painting", "teal painting",
    "black and white", "monochrome", "earth tones", "warm colors",
    "cool colors", "pastel colors", "vivid colors", "muted colors",
    "dark painting", "light painting", "golden", "silver",
    # Moods
    "serene", "dramatic", "melancholy", "joyful", "mysterious",
    "romantic", "peaceful", "intense", "ethereal", "whimsical",
    "haunting", "nostalgic", "playful", "somber", "vibrant",
    "contemplative", "dreamlike", "powerful", "gentle", "bold",
    # Objects & scenes
    "flowers", "trees", "mountains", "ocean", "sky", "clouds",
    "sunset", "sunrise", "night", "garden", "forest", "river",
    "lake", "beach", "village", "city", "street", "house",
    "church", "bridge", "boat", "ship", "horse", "dog", "cat",
    "bird", "fish", "fruit", "wine", "table", "chair", "window",
    "door", "mirror", "vase", "book", "candle", "lamp",
    # People
    "woman", "man", "child", "family", "couple", "dancer",
    "musician", "worker", "mother", "father", "girl", "boy",
    # Composition
    "close up", "wide view", "aerial view", "panoramic",
    "symmetrical", "asymmetrical", "geometric", "organic",
    "textured", "smooth", "detailed", "loose brushwork",
    "thick impasto", "thin glazes", "layered", "flat",
    # Time & place
    "spring", "summer", "autumn", "winter", "morning", "evening",
    "twilight", "moonlight", "rain", "snow", "fog", "storm",
    "rural", "urban", "tropical", "desert", "arctic",
    "Mediterranean", "Asian", "European", "American",
    # Qualities
    "beautiful", "elegant", "rustic", "primitive", "refined",
    "raw", "polished", "rough", "delicate", "massive",
    "tiny", "enormous", "intricate", "simple", "ornate",
    # Art movements
    "renaissance", "mannerism", "rococo", "neoclassical",
    "pre-raphaelite", "symbolism", "fauvism", "dadaism",
    "constructivism", "futurism", "suprematism", "de stijl",
    "abstract expressionism", "color field", "hard edge",
    "op art", "kinetic art", "land art", "conceptual art",
    # Famous artists (style references)
    "Monet style", "Van Gogh style", "Picasso style", "Rembrandt style",
    "Matisse style", "Klimt style", "Kandinsky style", "Rothko style",
    "Hopper style", "Warhol style", "Dali style", "Miro style",
    "Cezanne style", "Renoir style", "Degas style", "Turner style",
    # Materials & surfaces
    "canvas", "paper", "wood panel", "linen", "board",
    "copper", "glass", "fabric", "cardboard",
    # Emotions & feelings
    "love", "sadness", "joy", "anger", "fear",
    "hope", "loneliness", "freedom", "chaos", "harmony",
    "balance", "tension", "movement", "stillness", "energy",
    # Miscellaneous
    "vintage", "antique", "retro", "classic", "avant-garde",
    "experimental", "traditional", "innovative", "decorative", "functional",
]


# ─── PYIQA SCORING ───

def load_pyiqa_models(device):
    """Load TOPIQ-IAA and MUSIQ-AVA models."""
    import pyiqa

    print("  Loading TOPIQ-IAA model...")
    topiq = pyiqa.create_metric("topiq_iaa", device=device)
    print("  Loading MUSIQ-AVA model...")
    musiq = pyiqa.create_metric("musiq-ava", device=device)
    return topiq, musiq


@torch.no_grad()
def score_pyiqa(images, topiq_model, musiq_model, device):
    """Score a list of PIL images with pyiqa models.

    Returns (topiq_scores, musiq_scores, aesthetic2_scores) as numpy arrays.
    """
    # TOPIQ and MUSIQ handle their own preprocessing internally
    topiq_scores = []
    musiq_scores = []

    for img in images:
        try:
            topiq_raw = topiq_model(img).item()
            # TOPIQ-IAA (CFANet) outputs ~1-5 MOS scale → map to 0-100
            topiq_norm = np.clip((topiq_raw - 1.0) / 4.0 * 100.0, 0, 100)
            topiq_scores.append(topiq_norm)
        except Exception:
            topiq_scores.append(None)

        try:
            musiq_raw = musiq_model(img).item()
            # MUSIQ-AVA outputs ~1-10 AVA scale → map to 0-100
            musiq_norm = np.clip((musiq_raw - 1.0) / 9.0 * 100.0, 0, 100)
            musiq_scores.append(musiq_norm)
        except Exception:
            musiq_scores.append(None)

    topiq_arr = np.array([s if s is not None else 0.0 for s in topiq_scores])
    musiq_arr = np.array([s if s is not None else 0.0 for s in musiq_scores])

    # Blend 50/50
    aesthetic2 = topiq_arr * 0.5 + musiq_arr * 0.5

    # Track which had failures
    for i, (t, m) in enumerate(zip(topiq_scores, musiq_scores)):
        if t is None and m is None:
            aesthetic2[i] = None
        elif t is None:
            aesthetic2[i] = musiq_arr[i]
        elif m is None:
            aesthetic2[i] = topiq_arr[i]

    return topiq_arr, musiq_arr, aesthetic2


# ─── CLIP ZERO-SHOT TAGGING ───

@torch.no_grad()
def classify_zero_shot(model, tokenizer, image_features, prompts, device, top_k=1, threshold=0.05):
    """Classify images against a prompt dict using CLIP zero-shot.

    Returns list of lists of {tag, confidence} dicts.
    """
    labels = list(prompts.keys())
    texts = tokenizer(list(prompts.values())).to(device)
    text_feats = model.encode_text(texts)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)

    sims = image_features @ text_feats.T
    probs = sims.softmax(dim=-1).cpu().numpy()

    results = []
    for i in range(len(probs)):
        top_indices = probs[i].argsort()[::-1][:top_k]
        tags = [
            {"tag": labels[idx], "confidence": round(float(probs[i][idx]), 3)}
            for idx in top_indices
            if probs[i][idx] > threshold
        ]
        results.append(tags)

    return results


# ─── COLOR EXTRACTION ───

def extract_colors(img, n=5):
    """Extract dominant colors from an image using K-means.

    Returns list of {hex, rgb, pct} dicts sorted by cluster size.
    """
    from sklearn.cluster import MiniBatchKMeans

    small = img.resize((100, 100))
    pixels = np.array(small).reshape(-1, 3).astype(float)
    km = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42)
    km.fit(pixels)

    # Count pixels per cluster
    labels, counts = np.unique(km.labels_, return_counts=True)
    total = counts.sum()

    colors = []
    for idx in np.argsort(-counts):
        center = km.cluster_centers_[labels[idx]].astype(int)
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        pct = round(float(counts[idx]) / total * 100, 1)
        colors.append({"hex": hex_color, "rgb": [r, g, b], "pct": pct})

    return colors


def _hue_in_range(h, low, high):
    """Check if hue falls in range, handling wrap-around for red."""
    if low > high:
        return h >= low or h < high
    return low <= h < high


def categorize_colors(colors):
    """Map dominant colors to categorical tags.

    Returns list of unique tag strings (color names + tone tags).
    """
    tags = set()
    total_weight = sum(c["pct"] for c in colors)

    for c in colors:
        r, g, b = c["rgb"]
        # Convert to HLS
        h_norm, l_norm, s_norm = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        h = h_norm * 360  # 0-360
        s = s_norm * 100  # 0-100
        l = l_norm * 100  # 0-100

        weight = c["pct"] / total_weight if total_weight > 0 else 0

        # Only tag colors that take up meaningful space
        if weight < 0.1:
            continue

        # Color name from hue (only for sufficiently saturated colors)
        if s > 15 and 10 < l < 90:
            for name, (low, high) in COLOR_MAP.items():
                if _hue_in_range(h, low, high):
                    tags.add(name)
                    break

        # Tone tags
        if h < 65 or h > 310:
            tags.add("warm")
        if 170 <= h <= 310:
            tags.add("cool")
        if l < 30:
            tags.add("dark")
        if l > 70:
            tags.add("light")
        if s < 30:
            tags.add("muted")
        if s > 60:
            tags.add("vivid")

    return sorted(tags)


# ─── SEARCH VOCABULARY ───

@torch.no_grad()
def encode_search_vocab(model, tokenizer, device):
    """Encode search terms with CLIP text encoder.

    Returns dict of {term: float32_bytes}.
    """
    print(f"  Encoding {len(SEARCH_TERMS)} search terms...")
    vocab = {}
    batch_size = 64

    for i in range(0, len(SEARCH_TERMS), batch_size):
        batch_terms = SEARCH_TERMS[i:i + batch_size]
        prompts = [f"a painting of {t}" if not t.endswith("style") else f"a painting in the {t}" for t in batch_terms]
        tokens = tokenizer(prompts).to(device)
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        feats_np = feats.cpu().numpy()

        for j, term in enumerate(batch_terms):
            vocab[term] = feats_np[j].astype(np.float32).tobytes()

    print(f"  Encoded {len(vocab)} terms")
    return vocab


def push_search_vocab(vocab, redis_client):
    """Push encoded search vocabulary to Redis."""
    pipe = redis_client.pipeline()
    for term, emb_bytes in vocab.items():
        pipe.hset("stv:search_vocab", term, emb_bytes)
    pipe.execute()
    print(f"  Pushed {len(vocab)} search terms to Redis")


# ─── ENRICHMENT PIPELINE ───

def collect_unenriched(stores, force=False):
    """Collect paintings that have art_score but no aesthetic2.

    Returns list of (source, painting_id, painting_dict) tuples.
    """
    items = []
    for source, store in stores.items():
        for pid, p in store["listings"].items():
            if not p.get("images"):
                continue
            if p.get("art_score") is None:
                continue
            if not force and p.get("aesthetic2") is not None:
                continue
            items.append((source, pid, p))
    return items


@torch.no_grad()
def enrich_chunk(paintings, clip_model, preprocess, tokenizer, topiq_model, musiq_model, device, chunk_dir):
    """Enrich a chunk of paintings with pyiqa scores, CLIP tags, and colors.

    Returns list of enriched painting dicts.
    """
    # Download images
    image_map = download_chunk_images(paintings, chunk_dir)

    # Load and preprocess images — collect all valid ones first
    valid_indices = []
    pil_images = []
    clip_tensors = []

    for i, p in enumerate(paintings):
        pid = p.get("id", "")
        if pid not in image_map:
            continue
        try:
            img = Image.open(image_map[pid]).convert("RGB")
            tensor = preprocess(img)
            pil_images.append(img)
            clip_tensors.append(tensor)
            valid_indices.append(i)
        except Exception:
            continue

    if not clip_tensors:
        return []

    # Encode images with CLIP in batches
    all_features = []
    batch_size = 16
    for b in range(0, len(clip_tensors), batch_size):
        batch_tensor = torch.stack(clip_tensors[b:b + batch_size]).to(device)
        feats = clip_model.encode_image(batch_tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
        all_features.append(feats)

    image_features = torch.cat(all_features, dim=0)

    # pyiqa scoring
    print(f"    pyiqa scoring {len(pil_images)} images...")
    topiq_scores, musiq_scores, aesthetic2_scores = score_pyiqa(
        pil_images, topiq_model, musiq_model, device,
    )

    # CLIP zero-shot tags
    print(f"    CLIP tagging...")
    subjects = classify_zero_shot(clip_model, tokenizer, image_features, SUBJECT_PROMPTS, device, top_k=2, threshold=0.05)
    moods = classify_zero_shot(clip_model, tokenizer, image_features, MOOD_PROMPTS, device, top_k=1, threshold=0.05)
    medium_tags = classify_zero_shot(clip_model, tokenizer, image_features, MEDIUM_PROMPTS, device, top_k=1, threshold=0.05)

    # Color extraction
    print(f"    Extracting colors...")
    enriched = []
    for j, idx in enumerate(valid_indices):
        p = paintings[idx]

        # pyiqa scores
        p["topiq_score"] = round(float(topiq_scores[j]), 1)
        p["musiq_score"] = round(float(musiq_scores[j]), 1)
        p["aesthetic2"] = round(float(aesthetic2_scores[j]), 1)

        # CLIP tags
        p["subjects"] = subjects[j]
        p["moods"] = moods[j]
        p["medium_tags"] = medium_tags[j]

        # Colors
        colors = extract_colors(pil_images[j])
        p["colors"] = colors
        p["color_tags"] = categorize_colors(colors)

        enriched.append(p)

    return enriched


def enrich_paintings(limit=None, force=False, chunk_size=200, no_push=False):
    """Enrich scored paintings with additional metadata."""
    # Load stores
    stores = load_all_stores()
    if not stores:
        print("No data stores found in data/")
        return

    total_in_stores = sum(len(s["listings"]) for s in stores.values())
    print(f"Stores: {len(stores)} sources, {total_in_stores:,} total listings")

    # Collect unenriched
    items = collect_unenriched(stores, force=force)
    if not items:
        print("All scored paintings already enriched (use --force to re-enrich)")
        return

    if limit:
        items = items[:limit]

    print(f"To enrich: {len(items):,} paintings (chunk size: {chunk_size})")

    # Load models
    print("\nLoading models...")
    clip_model, preprocess, tokenizer, device = load_clip()
    topiq_model, musiq_model = load_pyiqa_models(device)

    # Connect to Redis (unless --no-push)
    redis_client = None
    redis_binary = None
    if not no_push:
        try:
            import redis
            from push import get_redis_url
            redis_url = get_redis_url()
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_binary = redis.from_url(redis_url, decode_responses=False)
            redis_client.ping()
            print("Connected to Redis")
        except Exception as e:
            print(f"Warning: Redis unavailable ({e}), enriching without push")
            redis_client = None
            redis_binary = None

    # Encode and push search vocabulary (once)
    if redis_binary:
        vocab = encode_search_vocab(clip_model, tokenizer, device)
        push_search_vocab(vocab, redis_binary)

    # Process in chunks
    THUMB_DIR.mkdir(exist_ok=True)
    total_enriched = 0
    total_pushed = 0
    num_chunks = math.ceil(len(items) / chunk_size)

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(items))
        chunk_items = items[start:end]
        chunk_paintings = [p for _, _, p in chunk_items]

        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_paintings)} paintings) ---")

        chunk_dir = THUMB_DIR / f"enrich_{chunk_idx}"
        enriched = enrich_chunk(
            chunk_paintings, clip_model, preprocess, tokenizer,
            topiq_model, musiq_model, device, chunk_dir,
        )

        if not enriched:
            print(f"  No images enriched in this chunk")
            continue

        total_enriched += len(enriched)

        # Push to Redis
        if redis_client:
            from push import push_paintings
            pushed, skipped = push_paintings(enriched, redis_client)
            total_pushed += pushed
            print(f"  Enriched {len(enriched)} -- pushed {pushed} to Redis "
                  f"({total_enriched}/{len(items)} total)")
        else:
            print(f"  Enriched {len(enriched)} ({total_enriched}/{len(items)} total)")

        # Write back to stores
        for source, pid, p in chunk_items:
            if p.get("aesthetic2") is not None:
                stores[source]["listings"][pid] = p

        # Save stores after each chunk
        sources_in_chunk = {source for source, _, _ in chunk_items}
        for source in sources_in_chunk:
            save_store(source, stores[source])

    # Summary
    if total_enriched:
        print(f"\n{'='*50}")
        print(f"  Enrichment complete")
        print(f"{'='*50}")
        print(f"  Enriched: {total_enriched}")
        if redis_client:
            print(f"  Pushed to Redis: {total_pushed}")

        # Show sample
        sample = [p for _, _, p in items[:5] if p.get("aesthetic2") is not None]
        if sample:
            print(f"\n  Sample enrichments:")
            for p in sample[:3]:
                subj = ", ".join(s["tag"] for s in (p.get("subjects") or []))
                mood = ", ".join(m["tag"] for m in (p.get("moods") or []))
                colors = ", ".join(p.get("color_tags") or [])
                print(f"    a2={p.get('aesthetic2', '?'):.1f} | {subj} | {mood} | {colors}")
                print(f"      {p['title'][:60]}")


def main():
    parser = argparse.ArgumentParser(description="Painting enrichment pipeline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Enrich only first N paintings")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich all paintings (ignore existing enrichment)")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Paintings per chunk (default: 200)")
    parser.add_argument("--no-push", action="store_true",
                        help="Enrich without pushing to Redis")
    args = parser.parse_args()

    enrich_paintings(
        limit=args.limit,
        force=args.force,
        chunk_size=args.chunk_size,
        no_push=args.no_push,
    )


if __name__ == "__main__":
    main()
