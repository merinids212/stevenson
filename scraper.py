"""
Craigslist California paintings-for-sale scraper.

Fetches painting listings from all Craigslist regions in California,
extracts structured data (JSON-LD + HTML), and saves to CSV/JSON.

Usage:
    python scraper.py
    python scraper.py --query "oil painting" --min-price 50 --max-price 5000
    python scraper.py --output paintings.csv --json
    python scraper.py --region losangeles  # single region only
"""

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup


# All Craigslist regions in California
CA_REGIONS = [
    "losangeles",
    "sfbay",
    "sandiego",
    "sacramento",
    "fresno",
    "bakersfield",
    "orangecounty",
    "inlandempire",
    "ventura",
    "santabarbara",
    "stockton",
    "modesto",
    "visalia",
    "merced",
    "monterey",
    "santacruz",
    "chico",
    "redding",
    "humboldt",
    "mendocino",
    "goldcountry",
    "susanville",
    "yubasutter",
    "palmsprings",
    "imperial",
    "slo",
    "hanford",
    "siskiyou",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

@dataclass
class Listing:
    title: str
    price: float | None
    url: str
    location: str
    latitude: float | None
    longitude: float | None
    images: list[str] = field(default_factory=list)
    posted: str = ""
    region: str = ""


def build_search_url(region: str, query: str, **filters) -> str:
    base = f"https://{region}.craigslist.org/search/sss"
    params = {"query": query, "sort": "date"}
    if filters.get("min_price"):
        params["min_price"] = filters["min_price"]
    if filters.get("max_price"):
        params["max_price"] = filters["max_price"]
    if filters.get("has_image"):
        params["hasPic"] = 1
    return f"{base}?{urlencode(params)}"


def fetch_page(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  Failed: {e}")
        return None


def parse_jsonld(soup: BeautifulSoup) -> list[dict]:
    """Extract listing data from JSON-LD script tags."""
    listings = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            continue

        items = []
        if isinstance(data, dict):
            items = data.get("itemListElement", [])
        elif isinstance(data, list):
            for d in data:
                if isinstance(d, dict):
                    items.extend(d.get("itemListElement", []))

        for entry in items:
            item = entry.get("item", entry)
            if item.get("@type") != "Product":
                continue
            listings.append(item)
    return listings


def parse_html_listings(soup: BeautifulSoup, region: str) -> dict[str, str]:
    """Build a map of title -> URL from static search result elements."""
    links = {}
    for li in soup.select("li.cl-static-search-result"):
        title = li.get("title", "").strip()
        a = li.find("a", href=True)
        if title and a:
            href = a["href"]
            if not href.startswith("http"):
                href = f"https://{region}.craigslist.org{href}"
            links[title] = href
    return links


def extract_listings(html: str, region: str) -> list[Listing]:
    """Parse a search results page into Listing objects."""
    soup = BeautifulSoup(html, "html.parser")
    jsonld_items = parse_jsonld(soup)
    html_links = parse_html_listings(soup, region)

    results = []
    for item in jsonld_items:
        title = item.get("name", "").strip()
        offers = item.get("offers", {})
        price_str = offers.get("price")
        price = float(price_str) if price_str else None

        # Location
        place = offers.get("availableAtOrFrom", {})
        addr = place.get("address", {})
        locality = addr.get("addressLocality", "")
        region_code = addr.get("addressRegion", "")
        location = f"{locality}, {region_code}".strip(", ")

        geo = place.get("geo", {})
        lat = geo.get("latitude")
        lng = geo.get("longitude")
        lat = float(lat) if lat else None
        lng = float(lng) if lng else None

        # Images
        images_raw = item.get("image", [])
        if isinstance(images_raw, str):
            images_raw = [images_raw]
        images = [img for img in images_raw if isinstance(img, str)]

        # URL — try matching from HTML links
        url = item.get("url", "")
        if not url:
            url = html_links.get(title, "")

        # Posted date
        posted = item.get("datePosted", "")

        results.append(
            Listing(
                title=title,
                price=price,
                url=url,
                location=location,
                latitude=lat,
                longitude=lng,
                images=images,
                posted=posted,
                region=region,
            )
        )
    return results


JUNK_KEYWORDS = [
    "book", "novel", "compressor", "scaffold", "mercedes", "benz", "scion",
    "ppe", "clean suit", "scissor lift", "golf course", "manufacturing",
    "equipment", "deck mat", "deck material", "coffee table", "cnc mill",
    "lathe", "grinder", "dryer", "forklift", "trailer", "truck", "car ",
    "sedan", "coupe", "suv", "honda", "toyota", "ford", "chevy", "bmw",
    "lexus", "audi", "nissan", "hyundai", "kia", "jeep", "dodge",
    "chrysler", "cadillac", "lincoln", "volvo", "subaru", "mazda",
    "foot cover", "face shield", "scaffolding", "pvc", "lumber",
    "plywood", "drywall", "concrete", "roofing", "flooring", "tile",
    "cabinet", "appliance", "washer", "refrigerator", "dishwasher",
    "microwave", "stove", "mattress",
]


def is_likely_painting(listing: Listing) -> bool:
    """Filter out listings that aren't actual paintings."""
    title = listing.title.lower()
    for kw in JUNK_KEYWORDS:
        if kw in title:
            return False
    # Must have "paint" somewhere in the title to be safe
    if "paint" not in title and "art" not in title and "canvas" not in title:
        return False
    return True


def scrape_region(
    region: str,
    query: str = "painting",
    min_price: int | None = None,
    max_price: int | None = None,
    has_image: bool = True,
) -> list[Listing]:
    """Scrape listings from a single Craigslist region."""
    url = build_search_url(
        region, query, min_price=min_price, max_price=max_price, has_image=has_image,
    )
    print(f"  {region}: {url}")

    html = fetch_page(url)
    if not html:
        return []

    return extract_listings(html, region)


def scrape(
    regions: list[str] | None = None,
    query: str = "painting",
    min_price: int | None = None,
    max_price: int | None = None,
    has_image: bool = True,
    delay: float = 1.0,
) -> list[Listing]:
    """Scrape listings from one or more Craigslist California regions."""
    if regions is None:
        regions = CA_REGIONS

    all_listings: list[Listing] = []
    for i, region in enumerate(regions):
        listings = scrape_region(
            region, query, min_price=min_price, max_price=max_price, has_image=has_image,
        )
        all_listings.extend(listings)

        # Be polite — delay between regions
        if i < len(regions) - 1 and delay > 0:
            time.sleep(delay)

    # Deduplicate by title
    seen: set[str] = set()
    unique: list[Listing] = []
    for listing in all_listings:
        if listing.title not in seen:
            seen.add(listing.title)
            unique.append(listing)

    # Filter out non-painting junk
    before = len(unique)
    unique = [l for l in unique if is_likely_painting(l)]
    junk = before - len(unique)
    print(f"\nTotal: {len(unique)} paintings across {len(regions)} regions "
          f"(filtered {junk} junk from {before} unique, {len(all_listings)} raw)")
    return unique


def save_csv(listings: list[Listing], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "price", "url", "location", "latitude", "longitude", "images", "posted", "region"])
        for l in listings:
            writer.writerow([
                l.title, l.price, l.url, l.location,
                l.latitude, l.longitude,
                "|".join(l.images), l.posted, l.region,
            ])
    print(f"Saved {len(listings)} listings to {path}")


def save_json(listings: list[Listing], path: Path):
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
    args = parser.parse_args()

    regions = [args.region] if args.region else None
    label = args.region or "all California"
    print(f'Scraping Craigslist {label} for "{args.query}"...\n')

    listings = scrape(
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


if __name__ == "__main__":
    main()
