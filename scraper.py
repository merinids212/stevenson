"""
Stevenson scraper orchestrator.

Each source accumulates into its own store under data/:
    data/ebay.json      — eBay listings, grows over time
    data/craigslist.json — Craigslist listings, grows over time

Each store is a JSON object:
    {
        "meta": { "last_run": "...", "total_runs": N, "total_scraped": N },
        "listings": { "eb:123": { ...fields, "first_seen": "...", "last_seen": "..." }, ... }
    }

The export step merges all stores into paintings.json for scoring/push.

Usage:
    python scraper.py --source ebay                # scrape eBay, accumulate into data/ebay.json
    python scraper.py --source craigslist           # scrape CL, accumulate into data/craigslist.json
    python scraper.py --source all                  # scrape both
    python scraper.py --export                      # merge stores → paintings.json (no scraping)
    python scraper.py --source ebay --export        # scrape eBay, then export
    python scraper.py --source ebay --push          # scrape, export, score-ready push
    python scraper.py --stats                       # show store stats

    # eBay options
    python scraper.py --source ebay --num-queries 6 --max-pages 3
    python scraper.py --source ebay --query "oil painting" "landscape painting"

    # Craigslist options
    python scraper.py --source craigslist --state ca ny
    python scraper.py --source craigslist --region losangeles
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from scrapers.craigslist import CraigslistScraper
from scrapers.ebay import EbayScraper

DATA_DIR = Path("data")


# ─── Store management ───

def load_store(source: str) -> dict:
    """Load a source's accumulated store, or create empty."""
    path = DATA_DIR / f"{source}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "meta": {
            "source": source,
            "created": datetime.now(timezone.utc).isoformat(),
            "last_run": None,
            "total_runs": 0,
            "total_scraped": 0,
        },
        "listings": {},
    }


def save_store(source: str, store: dict):
    """Save a source's store to disk."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{source}.json"
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    count = len(store["listings"])
    print(f"  Saved {count} listings to {path}")


def merge_into_store(store: dict, listings: list) -> tuple[int, int]:
    """Merge scraped listings into a store. Returns (new, updated) counts."""
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0
    updated_count = 0

    for listing in listings:
        data = asdict(listing) if hasattr(listing, '__dataclass_fields__') else dict(listing)
        pid = data["id"]

        if pid in store["listings"]:
            # Existing: update last_seen, refresh fields that might change (price)
            existing = store["listings"][pid]
            existing["last_seen"] = now
            if data.get("price") is not None:
                existing["price"] = data["price"]
            updated_count += 1
        else:
            # New: add with discovery timestamps
            data["first_seen"] = now
            data["last_seen"] = now
            store["listings"][pid] = data
            new_count += 1

    return new_count, updated_count


def print_store_stats():
    """Print stats for all stores."""
    DATA_DIR.mkdir(exist_ok=True)
    stores = sorted(DATA_DIR.glob("*.json"))
    if not stores:
        print("No stores found in data/")
        return

    total_all = 0
    for path in stores:
        store = json.loads(path.read_text())
        meta = store["meta"]
        listings = store["listings"]
        count = len(listings)
        total_all += count

        priced = [l for l in listings.values() if l.get("price") is not None]
        prices = [l["price"] for l in priced]
        with_images = sum(1 for l in listings.values() if l.get("images"))

        print(f"\n{'='*50}")
        print(f"  {meta.get('source', path.stem).upper()}")
        print(f"{'='*50}")
        print(f"  Listings:    {count:,}")
        print(f"  With images: {with_images:,}")
        print(f"  Runs:        {meta.get('total_runs', '?')}")
        print(f"  Last run:    {meta.get('last_run', 'never')}")
        if prices:
            prices.sort()
            print(f"  Price range: ${min(prices):,.0f} – ${max(prices):,.0f}")
            print(f"  Median:      ${prices[len(prices)//2]:,.0f}")

        # Age distribution
        seen_dates = []
        for l in listings.values():
            fs = l.get("first_seen")
            if fs:
                try:
                    seen_dates.append(datetime.fromisoformat(fs))
                except (ValueError, TypeError):
                    pass
        if seen_dates:
            oldest = min(seen_dates).strftime("%Y-%m-%d")
            newest = max(seen_dates).strftime("%Y-%m-%d")
            print(f"  First seen:  {oldest} → {newest}")

    print(f"\n  Total across all stores: {total_all:,}")


# ─── Export: merge stores into paintings.json ───

def export_paintings(output: Path):
    """Merge all stores into a flat paintings.json for scoring/push."""
    DATA_DIR.mkdir(exist_ok=True)
    all_listings = []

    for path in sorted(DATA_DIR.glob("*.json")):
        store = json.loads(path.read_text())
        source = store["meta"].get("source", path.stem)
        listings = list(store["listings"].values())
        # Only include listings with images
        listings = [l for l in listings if l.get("images")]
        print(f"  {source}: {len(listings)} listings with images")
        all_listings.extend(listings)

    if not all_listings:
        print("No listings to export.")
        return []

    # Sort by first_seen descending (newest first)
    all_listings.sort(key=lambda l: l.get("first_seen", ""), reverse=True)

    output.write_text(json.dumps(all_listings, indent=2, ensure_ascii=False))
    print(f"\nExported {len(all_listings)} paintings to {output}")

    # Price stats
    priced = [l for l in all_listings if l.get("price") is not None]
    if priced:
        prices = sorted(l["price"] for l in priced)
        print(f"  Price range: ${min(prices):,.0f} – ${max(prices):,.0f}")
        print(f"  Median: ${prices[len(prices)//2]:,.0f}")

    return all_listings


# ─── CSV export ───

def save_csv(listings: list[dict], path: Path):
    fields = [
        "id", "source", "title", "price", "url", "location",
        "latitude", "longitude", "images", "posted", "region", "state",
        "first_seen", "last_seen",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for l in listings:
            images = l.get("images", [])
            if isinstance(images, list):
                images = "|".join(images)
            writer.writerow([l.get(k, "") for k in fields[:8]] + [images] +
                            [l.get(k, "") for k in fields[9:]])
    print(f"Saved {len(listings)} listings to {path}")


# ─── Source runners ───

def run_craigslist(args, known_ids: set[str] | None = None):
    """Generator: yields batches of CL listings per region."""
    regions = [args.region] if args.region else None
    states = [s.lower() for s in args.state] if args.state else None

    cl_queries = getattr(args, "cl_queries", None)
    num_cl_queries = getattr(args, "num_cl_queries", None)
    cl_max_pages = getattr(args, "cl_max_pages", 1)

    if args.region:
        label = args.region
    elif args.state:
        label = ", ".join(s.upper() for s in states)
    else:
        label = "all US"
    print(f'Scraping Craigslist {label}...\n', flush=True)

    scraper = CraigslistScraper()
    yield from scraper.scrape(
        regions=regions,
        states=states,
        query=args.query,
        queries=cl_queries,
        num_queries=num_cl_queries,
        min_price=args.min_price,
        max_price=args.max_price,
        has_image=not args.no_image,
        delay=args.delay,
        max_pages=cl_max_pages,
        known_ids=known_ids,
    )


def run_ebay(args, known_ids: set[str]):
    """Generator: yields batches of eBay listings per query."""
    queries = args.ebay_queries if args.ebay_queries else None
    print(f"Scraping eBay for paintings...\n", flush=True)

    scraper = EbayScraper()
    yield from scraper.scrape(
        queries=queries,
        num_queries=args.num_queries,
        min_price=args.min_price,
        max_price=args.max_price,
        max_pages=args.max_pages,
        delay=args.delay_ebay,
        known_ids=known_ids,
    )


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="Scrape paintings from Craigslist and eBay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--source", choices=["craigslist", "ebay", "all"],
                        default=None, help="Source to scrape")
    parser.add_argument("--export", action="store_true",
                        help="Export merged stores to paintings.json")
    parser.add_argument("--stats", action="store_true",
                        help="Print store stats and exit")
    parser.add_argument("--push", action="store_true",
                        help="Push paintings.json to Redis after export")

    # Shared filters
    parser.add_argument("-q", "--query", default="painting",
                        help="Search query for Craigslist (default: painting)")
    parser.add_argument("--min-price", type=int, default=None)
    parser.add_argument("--max-price", type=int, default=None)

    # Craigslist
    parser.add_argument("--no-image", action="store_true",
                        help="Include CL listings without images")
    parser.add_argument("--region", default=None,
                        help="Single CL region (e.g. losangeles)")
    parser.add_argument("--state", nargs="+", default=None,
                        help="CL state(s) (e.g. ca ny tx fl)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between CL regions (default: 1.0)")
    parser.add_argument("--cl-queries", nargs="+", default=None,
                        help="Explicit CL queries (overrides --query)")
    parser.add_argument("--num-cl-queries", type=int, default=None,
                        help="Pick N random queries from CL query pool")
    parser.add_argument("--cl-max-pages", type=int, default=3,
                        help="Max CL pages per region/query (default: 3)")

    # eBay
    parser.add_argument("--ebay-queries", nargs="+", default=None,
                        help="Explicit eBay queries (overrides random selection)")
    parser.add_argument("--num-queries", type=int, default=4,
                        help="Number of random queries per eBay run (default: 4)")
    parser.add_argument("--delay-ebay", type=float, default=2.0,
                        help="Base delay between eBay pages (default: 2.0)")
    parser.add_argument("--max-pages", type=int, default=5,
                        help="Max eBay pages per query (default: 5)")

    args = parser.parse_args()

    # Stats mode
    if args.stats:
        print_store_stats()
        return

    # Need either --source or --export
    if not args.source and not args.export:
        parser.print_help()
        sys.exit(1)

    now = datetime.now(timezone.utc).isoformat()

    # ─── Scrape phase ───
    if args.source in ("craigslist", "all"):
        store = load_store("craigslist")
        known_ids = set(store["listings"].keys())
        print(f"  Store has {len(known_ids)} existing CL listings\n", flush=True)
        total_new = total_updated = 0
        for batch in run_craigslist(args, known_ids=known_ids):
            new, updated = merge_into_store(store, batch)
            total_new += new
            total_updated += updated
            store["meta"]["last_run"] = now
            store["meta"]["total_scraped"] += len(batch)
            save_store("craigslist", store)
        if total_new or total_updated:
            store["meta"]["total_runs"] += 1
            save_store("craigslist", store)
            print(f"  Craigslist: {total_new} new, {total_updated} updated, "
                  f"{len(store['listings'])} total in store\n", flush=True)

    if args.source in ("ebay", "all"):
        store = load_store("ebay")
        known_ids = set(store["listings"].keys())
        print(f"  Store has {len(known_ids)} existing listings\n", flush=True)
        total_new = total_updated = 0
        for batch in run_ebay(args, known_ids):
            new, updated = merge_into_store(store, batch)
            total_new += new
            total_updated += updated
            store["meta"]["last_run"] = now
            store["meta"]["total_scraped"] += len(batch)
            save_store("ebay", store)
        if total_new or total_updated:
            store["meta"]["total_runs"] += 1
            save_store("ebay", store)
            print(f"  eBay: {total_new} new, {total_updated} updated, "
                  f"{len(store['listings'])} total in store\n", flush=True)

    # ─── Export phase ───
    if args.export or args.push:
        print("\nExporting all stores to paintings.json...")
        output = Path("paintings.json")
        all_listings = export_paintings(output)

        if all_listings:
            save_csv(all_listings, output.with_suffix(".csv"))

    # ─── Push phase ───
    if args.push:
        json_path = Path("paintings.json")
        if not json_path.exists():
            print("No paintings.json to push")
            sys.exit(1)
        from push import push
        push(str(json_path), flush=True)


if __name__ == "__main__":
    main()
