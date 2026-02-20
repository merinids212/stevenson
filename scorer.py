"""
CLIP-based painting quality scorer.

Downloads painting thumbnails, runs CLIP embeddings + aesthetic scoring,
and enriches paintings.json with quality metrics.

Usage:
    python scorer.py                    # Score all paintings
    python scorer.py --limit 50         # Score first 50 only (for testing)
    python scorer.py --skip-download    # Reuse already downloaded images
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

# Lazy imports for CLIP
open_clip = None
tokenizer_fn = None


def load_clip():
    """Load CLIP model (ViT-B/32) and preprocessing."""
    global open_clip
    import open_clip as oc
    open_clip = oc

    print("Loading CLIP model (ViT-B/32)...")
    model, _, preprocess = oc.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = oc.get_tokenizer("ViT-B-32")
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"  Model loaded on {device}")

    return model, preprocess, tokenizer, device


# ─── AESTHETIC PREDICTOR ───
# Simple linear aesthetic predictor trained on LAION aesthetics data
# This is a lightweight MLP that maps CLIP embeddings → aesthetic score (1-10)

class AestheticPredictor(torch.nn.Module):
    """Lightweight aesthetic MLP on top of CLIP features."""
    def __init__(self, input_dim=512):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x)


def load_aesthetic_model(device):
    """Load or initialize the aesthetic predictor."""
    model = AestheticPredictor(input_dim=512).to(device)
    weights_path = Path("aesthetic_model.pth")

    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print("  Loaded aesthetic model weights")
    else:
        # No pretrained weights available — use CLIP text prompts as proxy
        print("  No aesthetic weights found — using CLIP prompt-based scoring")
        return None

    model.eval()
    return model


# ─── CLIP PROMPT-BASED QUALITY SCORING ───
# When we don't have a trained aesthetic model, use CLIP text-image similarity
# to score quality against curated prompts

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

    # Average similarity to high vs low prompts
    high_sim = (image_features @ high_feats.T).mean(dim=-1)
    low_sim = (image_features @ low_feats.T).mean(dim=-1)

    # Convert to 0-100 score
    # Higher high_sim and lower low_sim = better
    raw = high_sim - low_sim
    # Normalize to roughly 0-100 range
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

    sims = image_features @ text_feats.T  # (N, num_styles)
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

    # 2. Load CLIP
    print("\n2. Loading CLIP...")
    model, preprocess, tokenizer, device = load_clip()

    # 3. Encode all images
    print(f"\n3. Encoding {len(items)} images...")
    all_features = []
    valid_indices = []

    batch_size = 32
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

    # 4. Quality scoring
    print("\n4. Computing quality scores...")
    quality_scores = compute_prompt_scores(model, tokenizer, image_features, device)

    # 5. Style classification
    print("\n5. Classifying styles...")
    styles = classify_styles(model, tokenizer, image_features, device)

    # 6. Compute visual similarity (find unique vs derivative)
    print("\n6. Computing uniqueness scores...")
    sim_matrix = (image_features @ image_features.T).cpu().numpy()
    np.fill_diagonal(sim_matrix, 0)
    # Uniqueness = inverse of average similarity to other paintings
    avg_similarity = sim_matrix.mean(axis=1)
    uniqueness = ((1 - avg_similarity) * 100).clip(0, 100)

    # 7. Assign scores back to paintings
    print("\n7. Enriching painting data...")
    for j, idx in enumerate(valid_indices):
        items[idx]["quality_score"] = round(float(quality_scores[j]), 1)
        items[idx]["clip_styles"] = styles[j]
        items[idx]["uniqueness"] = round(float(uniqueness[j]), 1)

        # Composite "art score" = weighted blend
        q = quality_scores[j]
        u = uniqueness[j]
        items[idx]["art_score"] = round(float(q * 0.7 + u * 0.3), 1)

    # Stats
    scored = [p for p in items if "art_score" in p]
    if scored:
        scores = [p["art_score"] for p in scored]
        print(f"\n--- Scoring complete ---")
        print(f"Scored: {len(scored)} paintings")
        print(f"Art score range: {min(scores):.1f} – {max(scores):.1f}")
        print(f"Median art score: {sorted(scores)[len(scores)//2]:.1f}")

        # Top 5
        top = sorted(scored, key=lambda p: p["art_score"], reverse=True)[:5]
        print(f"\nTop 5:")
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
