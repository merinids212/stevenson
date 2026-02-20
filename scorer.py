"""
CLIP-based painting quality scorer with LAION aesthetic predictor.

Downloads painting thumbnails, runs CLIP ViT-L/14 embeddings,
scores aesthetics with the LAION improved-aesthetic-predictor MLP,
parses artist names, computes value scores, and enriches paintings.json.

Usage:
    python scorer.py                    # Score all paintings
    python scorer.py --limit 50         # Score first 50 only (for testing)
    python scorer.py --skip-download    # Reuse already downloaded images
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

# Lazy imports for CLIP
open_clip = None


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
    model = model.to(device)
    print(f"  Model loaded on {device}")

    return model, preprocess, tokenizer, device


# ─── LAION AESTHETIC MLP ───
# MLP from christophschuhmann/improved-aesthetic-predictor
# Trained on LAION aesthetics data, outputs calibrated 1-10 scores
# Architecture must match exactly (no ReLU activations — they're commented out in the original)

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
    "cal", "so", "la",  # abbreviations for California, Southern, Los Angeles
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

# Words that signal end of a name (common in art listing titles)
NAME_STOP_WORDS = {
    # media & surface
    "oil", "on", "canvas", "acrylic", "watercolor", "painting", "print",
    "lithograph", "etching", "engraving", "serigraph", "giclee",
    # condition & framing
    "original", "framed", "signed", "large", "small", "vintage", "antique",
    "unfinished", "matted", "mounted",
    # art terms
    "art", "artwork", "piece", "gallery", "museum", "rare", "proof",
    "style", "scene", "geometric", "impressionist", "surreal",
    # adjectives
    "beautiful", "stunning", "gorgeous", "amazing", "exceptional", "fine",
    "great", "new", "old", "modern", "contemporary",
    # genres
    "abstract", "landscape", "portrait", "still", "life", "floral",
    # prepositions & connectors
    "with", "and", "the", "from", "for", "in", "of", "at", "to", "or",
    # misc listing words
    "mid", "century", "listed", "numbered", "hand", "painted", "wrap",
    # animals & objects (common false captures)
    "tiger", "cubs", "cat", "cats", "dog", "horse", "flower", "flowers",
    "raccoon", "fish", "clown", "bird", "pirate", "ship", "street",
    # places
    "hong", "kong", "paris", "east", "coast", "native", "american",
    # meta
    "who", "has", "work", "archetype", "best", "offer",
    # misc false positives
    "jewish", "men", "boy", "blue", "fisherman", "lake", "nostalgic",
    "number",
}

# Full-name rejects: these get captured as names but aren't
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
    """Trim leading prefixes and trailing stop words from a captured name."""
    words = name.split()
    # Strip leading qualifiers: "California Artist Susie Cartt" → "Susie Cartt"
    while words and words[0].lower() in _NAME_PREFIX_STRIP:
        words.pop(0)
    # Strip trailing stop words
    while words and words[-1].lower() in NAME_STOP_WORDS:
        words.pop()
    return " ".join(words)


def _valid_name(name):
    """Check if a cleaned name looks like an actual person name."""
    if not name or len(name) < 3:
        return False
    if name.lower() in NAME_REJECTS:
        return False
    # Reject single very short words (TAR, Cal, etc.) — likely abbreviations
    words = name.split()
    if len(words) == 1 and len(name) < 4:
        return False
    return True


def parse_artist(title):
    """Extract artist name from painting title.

    Returns (artist_name, confidence) or (None, None).
    """
    if not title:
        return None, None

    t = title.strip()

    # Pattern 1: "... by [Name]" — highest confidence
    m = re.search(r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})', t)
    if m:
        name = _clean_name(m.group(1).strip())
        first_word = name.split()[0].lower() if name else ""
        if first_word not in BY_EXCLUSIONS and _valid_name(name):
            return name, "high"

    # Pattern 2: "Signed [Name]" — max 2 words, name must start uppercase
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

    # Pattern 3: "Artist [Name]" or "Artist: [Name]" — name must start uppercase
    m = re.search(
        r'\b[Aa][Rr][Tt][Ii][Ss][Tt]:?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})',
        t,
    )
    if m:
        name = _clean_name(m.group(1).strip())
        if _valid_name(name):
            return name, "medium"

    # Pattern 4: Known famous names — word boundary match to avoid substrings
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
    """Compute gem/value score: high art + low price = high value.

    Formula: (art_score / 100) / log(price + 1) * 100
    """
    if art_score is None or price is None or price <= 0:
        return None
    return round((art_score / 100) / math.log(price + 1) * 100, 1)


# ─── IMAGE DOWNLOAD ───

THUMB_DIR = Path("thumbs")


def download_images(paintings, limit=None):
    """Download painting thumbnails."""
    THUMB_DIR.mkdir(exist_ok=True)
    items = paintings[:limit] if limit else paintings
    downloaded = 0
    skipped = 0

    for i, p in enumerate(items):
        if not p.get("images"):
            continue
        img_path = THUMB_DIR / f"{i}.jpg"
        if img_path.exists():
            skipped += 1
            continue

        url = p["images"][0]
        try:
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0"
            })
            resp.raise_for_status()
            img_path.write_bytes(resp.content)
            downloaded += 1
        except Exception:
            continue

        if (downloaded + skipped) % 100 == 0:
            print(f"  Progress: {downloaded + skipped}/{len(items)}")

    print(f"  Downloaded {downloaded}, skipped {skipped} existing")


# ─── MAIN PIPELINE ───

@torch.no_grad()
def score_paintings(paintings, limit=None, skip_download=False):
    """Run full scoring pipeline."""
    items = paintings[:limit] if limit else paintings

    # 1. Download images
    if not skip_download:
        print("\n1. Downloading images...")
        download_images(items)
    else:
        print("\n1. Skipping download (--skip-download)")

    # 2. Load CLIP (ViT-L/14)
    print("\n2. Loading CLIP (ViT-L/14)...")
    model, preprocess, tokenizer, device = load_clip()

    # 3. Load LAION aesthetic predictor
    print("\n3. Loading LAION aesthetic predictor...")
    aesthetic_model = load_aesthetic_model(device)

    # 4. Encode all images
    print(f"\n4. Encoding {len(items)} images...")
    all_features = []
    valid_indices = []

    batch_size = 16  # Smaller batches for ViT-L/14
    batch_images = []
    batch_indices = []

    for i in range(len(items)):
        img_path = THUMB_DIR / f"{i}.jpg"
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = preprocess(img)
            batch_images.append(tensor)
            batch_indices.append(i)
        except Exception:
            continue

        if len(batch_images) >= batch_size:
            batch_tensor = torch.stack(batch_images).to(device)
            feats = model.encode_image(batch_tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_features.append(feats)
            valid_indices.extend(batch_indices)
            batch_images = []
            batch_indices = []

            if len(valid_indices) % 200 == 0:
                print(f"  Encoded {len(valid_indices)} images...")

    # Remaining batch
    if batch_images:
        batch_tensor = torch.stack(batch_images).to(device)
        feats = model.encode_image(batch_tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
        all_features.append(feats)
        valid_indices.extend(batch_indices)

    if not all_features:
        print("No images to score!")
        return items

    image_features = torch.cat(all_features, dim=0)
    print(f"  Encoded {len(valid_indices)} images total")

    # 5. LAION aesthetic scoring
    print("\n5. Computing LAION aesthetic scores...")
    aesthetic_scores = aesthetic_model(image_features).squeeze(-1).cpu().numpy()
    print(f"  Aesthetic range: {aesthetic_scores.min():.2f} – {aesthetic_scores.max():.2f}")

    # 6. Prompt-based quality scoring
    print("\n6. Computing prompt quality scores...")
    prompt_scores = compute_prompt_scores(model, tokenizer, image_features, device)

    # 7. Style classification
    print("\n7. Classifying styles...")
    styles = classify_styles(model, tokenizer, image_features, device)

    # 8. Uniqueness
    print("\n8. Computing uniqueness scores...")
    sim_matrix = (image_features @ image_features.T).cpu().numpy()
    np.fill_diagonal(sim_matrix, 0)
    avg_similarity = sim_matrix.mean(axis=1)
    uniqueness = ((1 - avg_similarity) * 100).clip(0, 100)

    # 9. Parse artist names
    print("\n9. Parsing artist names...")
    artist_count = 0
    for p in items:
        artist, confidence = parse_artist(p.get("title", ""))
        if artist:
            p["artist"] = artist
            p["artist_confidence"] = confidence
            artist_count += 1
        else:
            p["artist"] = None
            p["artist_confidence"] = None
    print(f"  Found {artist_count} artist names")

    # 10. Enrich painting data
    print("\n10. Enriching painting data...")
    # Normalize LAION scores to 0-100 for composite
    aes_min, aes_max = float(aesthetic_scores.min()), float(aesthetic_scores.max())
    aes_range = aes_max - aes_min if aes_max > aes_min else 1.0
    laion_norm = (aesthetic_scores - aes_min) / aes_range * 100

    for j, idx in enumerate(valid_indices):
        items[idx]["aesthetic_score"] = round(float(aesthetic_scores[j]), 2)
        items[idx]["quality_score"] = round(float(prompt_scores[j]), 1)
        items[idx]["clip_styles"] = styles[j]
        items[idx]["uniqueness"] = round(float(uniqueness[j]), 1)

        # Composite art score: LAION 50% + prompt quality 20% + uniqueness 30%
        ln = laion_norm[j]
        pq = prompt_scores[j]
        u = uniqueness[j]
        items[idx]["art_score"] = round(float(ln * 0.5 + pq * 0.2 + u * 0.3), 1)

        # Value score
        items[idx]["value_score"] = compute_value_score(
            items[idx]["art_score"],
            items[idx].get("price"),
        )

    # Stats
    scored = [p for p in items if p.get("art_score") is not None]
    if scored:
        scores = [p["art_score"] for p in scored]
        aes = [p["aesthetic_score"] for p in scored if p.get("aesthetic_score") is not None]
        print(f"\n--- Scoring complete ---")
        print(f"Scored: {len(scored)} paintings")
        print(f"Art score range: {min(scores):.1f} – {max(scores):.1f}")
        print(f"Median art score: {sorted(scores)[len(scores)//2]:.1f}")
        if aes:
            print(f"LAION aesthetic range: {min(aes):.2f} – {max(aes):.2f}")
            print(f"Median LAION aesthetic: {sorted(aes)[len(aes)//2]:.2f}")

        artists = [p for p in items if p.get("artist")]
        print(f"Artists found: {len(artists)}")

        gems = [p for p in scored if p.get("value_score") is not None]
        if gems:
            top_gems = sorted(gems, key=lambda p: p["value_score"], reverse=True)[:5]
            print(f"\nTop 5 Gems (value score):")
            for p in top_gems:
                print(f"  {p['value_score']:.1f} | art:{p['art_score']:.1f} | ${p.get('price', '?')} | {p['title'][:50]}")

        top = sorted(scored, key=lambda p: p["art_score"], reverse=True)[:5]
        print(f"\nTop 5 (art score):")
        for p in top:
            print(f"  {p['art_score']:.1f} | ${p.get('price', '?')} | {p['title'][:60]}")

    return items


def main():
    parser = argparse.ArgumentParser(description="CLIP-based painting scorer")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of paintings to score")
    parser.add_argument("--skip-download", action="store_true", help="Skip image download")
    parser.add_argument("-i", "--input", default="paintings.json", help="Input JSON")
    parser.add_argument("-o", "--output", default="paintings.json", help="Output JSON (overwrites input by default)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    paintings = json.loads(input_path.read_text())
    print(f"Loaded {len(paintings)} paintings from {input_path}")

    score_paintings(paintings, limit=args.limit, skip_download=args.skip_download)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(paintings, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(paintings)} paintings to {output_path}")


if __name__ == "__main__":
    main()
