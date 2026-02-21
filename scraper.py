"""
Stevenson scraper orchestrator.

Runs configured scrapers, deduplicates results, and saves to CSV/JSON.
Optionally pushes results to Redis.

Usage:
    python scraper.py
    python scraper.py --query "oil painting" --min-price 50 --max-price 5000
    python scraper.py --output paintings.csv --json
    python scraper.py --region losangeles --push
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from scrapers.craigslist import CraigslistScraper


def save_csv(listings, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "source", "title", "price", "url", "location",
            "latitude", "longitude", "images", "posted", "region",
        ])
        for l in listings:
            writer.writerow([
                l.id, l.source, l.title, l.price, l.url, l.location,
                l.latitude, l.longitude,
                "|".join(l.images), l.posted, l.region,
            ])
    print(f"Saved {len(listings)} listings to {path}")


def save_json(listings, path: Path):
    data = [asdict(l) for l in listings]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved {len(listings)} listings to {path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Craigslist California for paintings")
    parser.add_argument("-q", "--query", default="painting", help="Search query (default: painting)")
    parser.add_argument("--min-price", type=int, default=None, help="Minimum price filter")
    parser.add_argument("--max-price", type=int, default=None, help="Maximum price filter")
    parser.add_argument("--no-image", action="store_true", help="Include listings without images")
    parser.add_argument("-o", "--output", default="paintings.csv", help="Output CSV file (default: paintings.csv)")
    parser.add_argument("--json", action="store_true", help="Also save as JSON")
    parser.add_argument("--region", default=None, help="Single region to scrape (default: all California)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between regions in seconds (default: 1.0)")
    parser.add_argument("--push", action="store_true", help="Push results to Redis after saving")
    args = parser.parse_args()

    regions = [args.region] if args.region else None
    label = args.region or "all California"
    print(f'Scraping Craigslist {label} for "{args.query}"...\n')

    scraper = CraigslistScraper()
    listings = scraper.scrape(
        regions=regions,
        query=args.query,
        min_price=args.min_price,
        max_price=args.max_price,
        has_image=not args.no_image,
        delay=args.delay,
    )

    if not listings:
        print("No listings found.")
        sys.exit(1)

    # Price stats
    priced = [l for l in listings if l.price is not None]
    if priced:
        prices = [l.price for l in priced]
        print(f"\n--- {len(listings)} listings scraped ---")
        print(f"Price range: ${min(prices):,.0f} – ${max(prices):,.0f}")
        print(f"Median: ${sorted(prices)[len(prices)//2]:,.0f}")
    else:
        print(f"\n--- {len(listings)} listings scraped (no prices) ---")

    out = Path(args.output)
    save_csv(listings, out)
    if args.json:
        save_json(listings, out.with_suffix(".json"))

    if args.push:
        json_path = out.with_suffix(".json")
        if not json_path.exists():
            save_json(listings, json_path)
        from push import push
        push(str(json_path), flush=True)


if __name__ == "__main__":
    main()
