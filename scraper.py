"""
Craigslist LA paintings-for-sale scraper.

Fetches painting listings from Craigslist Los Angeles,
extracts structured data (JSON-LD + HTML), and saves to CSV.

Usage:
    python scraper.py
    python scraper.py --query "oil painting" --min-price 50 --max-price 5000
    python scraper.py --output paintings.csv --json
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://losangeles.craigslist.org/search/sss"
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


def build_search_url(query: str, **filters) -> str:
    params = {"query": query, "sort": "date"}
    if filters.get("min_price"):
        params["min_price"] = filters["min_price"]
    if filters.get("max_price"):
        params["max_price"] = filters["max_price"]
    if filters.get("has_image"):
        params["hasPic"] = 1
    return f"{BASE_URL}?{urlencode(params)}"


def fetch_page(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


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


def parse_html_listings(soup: BeautifulSoup) -> dict[str, str]:
    """Build a map of title -> URL from static search result elements."""
    links = {}
    for li in soup.select("li.cl-static-search-result"):
        title = li.get("title", "").strip()
        a = li.find("a", href=True)
        if title and a:
            href = a["href"]
            if not href.startswith("http"):
                href = f"https://losangeles.craigslist.org{href}"
            links[title] = href
    return links


def extract_listings(html: str) -> list[Listing]:
    """Parse a search results page into Listing objects."""
    soup = BeautifulSoup(html, "html.parser")
    jsonld_items = parse_jsonld(soup)
    html_links = parse_html_listings(soup)

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
        region = addr.get("addressRegion", "")
        location = f"{locality}, {region}".strip(", ")

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
            )
        )
    return results


def scrape(
    query: str = "painting",
    min_price: int | None = None,
    max_price: int | None = None,
    has_image: bool = True,
) -> list[Listing]:
    """Scrape listings from Craigslist LA search results."""
    url = build_search_url(
        query, min_price=min_price, max_price=max_price, has_image=has_image,
    )
    print(f"Fetching {url}")

    html = fetch_page(url)
    listings = extract_listings(html)

    # Deduplicate by title
    seen: set[str] = set()
    unique: list[Listing] = []
    for listing in listings:
        if listing.title not in seen:
            seen.add(listing.title)
            unique.append(listing)

    print(f"Found {len(unique)} unique listings (from {len(listings)} total)")
    return unique


def save_csv(listings: list[Listing], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "price", "url", "location", "latitude", "longitude", "images", "posted"])
        for l in listings:
            writer.writerow([
                l.title, l.price, l.url, l.location,
                l.latitude, l.longitude,
                "|".join(l.images), l.posted,
            ])
    print(f"Saved {len(listings)} listings to {path}")


def save_json(listings: list[Listing], path: Path):
    data = [asdict(l) for l in listings]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved {len(listings)} listings to {path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Craigslist LA for paintings")
    parser.add_argument("-q", "--query", default="painting", help="Search query (default: painting)")
    parser.add_argument("--min-price", type=int, default=None, help="Minimum price filter")
    parser.add_argument("--max-price", type=int, default=None, help="Maximum price filter")
    parser.add_argument("--no-image", action="store_true", help="Include listings without images")
    parser.add_argument("-o", "--output", default="paintings.csv", help="Output CSV file (default: paintings.csv)")
    parser.add_argument("--json", action="store_true", help="Also save as JSON")
    args = parser.parse_args()

    print(f'Scraping Craigslist LA for "{args.query}"...\n')
    listings = scrape(
        query=args.query,
        min_price=args.min_price,
        max_price=args.max_price,
        has_image=not args.no_image,
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
