"""Craigslist California paintings scraper."""

import json
import re
import time
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from .base import BaseScraper, Listing


class CraigslistScraper(BaseScraper):
    source = "cl"

    CA_REGIONS = [
        "losangeles", "sfbay", "sandiego", "sacramento", "fresno",
        "bakersfield", "orangecounty", "inlandempire", "ventura",
        "santabarbara", "stockton", "modesto", "visalia", "merced",
        "monterey", "santacruz", "chico", "redding", "humboldt",
        "mendocino", "goldcountry", "susanville", "yubasutter",
        "palmsprings", "imperial", "slo", "hanford", "siskiyou",
    ]

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

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

    def _extract_id(self, url: str) -> str:
        m = re.search(r"/(\d+)\.html", url)
        return m.group(1) if m else ""

    def _build_search_url(self, region: str, query: str, **filters) -> str:
        base = f"https://{region}.craigslist.org/search/sss"
        params = {"query": query, "sort": "date"}
        if filters.get("min_price"):
            params["min_price"] = filters["min_price"]
        if filters.get("max_price"):
            params["max_price"] = filters["max_price"]
        if filters.get("has_image"):
            params["hasPic"] = 1
        return f"{base}?{urlencode(params)}"

    def _fetch_page(self, url: str) -> str | None:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"  Failed: {e}")
            return None

    def _parse_jsonld(self, soup: BeautifulSoup) -> list[dict]:
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

    def _parse_html_listings(self, soup: BeautifulSoup, region: str) -> dict[str, str]:
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

    def _extract_listings(self, html: str, region: str) -> list[Listing]:
        soup = BeautifulSoup(html, "html.parser")
        jsonld_items = self._parse_jsonld(soup)
        html_links = self._parse_html_listings(soup, region)

        results = []
        for item in jsonld_items:
            title = item.get("name", "").strip()
            offers = item.get("offers", {})
            price_str = offers.get("price")
            price = float(price_str) if price_str else None

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

            images_raw = item.get("image", [])
            if isinstance(images_raw, str):
                images_raw = [images_raw]
            images = [img for img in images_raw if isinstance(img, str)]

            url = item.get("url", "")
            if not url:
                url = html_links.get(title, "")

            posted = item.get("datePosted", "")
            ext_id = self._extract_id(url)

            results.append(
                Listing(
                    id=f"cl:{ext_id}" if ext_id else "",
                    source="cl",
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

    def _is_likely_painting(self, listing: Listing) -> bool:
        title = listing.title.lower()
        for kw in self.JUNK_KEYWORDS:
            if kw in title:
                return False
        if "paint" not in title and "art" not in title and "canvas" not in title:
            return False
        return True

    def _scrape_region(
        self, region: str, query: str = "painting",
        min_price: int | None = None, max_price: int | None = None,
        has_image: bool = True,
    ) -> list[Listing]:
        url = self._build_search_url(
            region, query, min_price=min_price, max_price=max_price, has_image=has_image,
        )
        print(f"  {region}: {url}")
        html = self._fetch_page(url)
        if not html:
            return []
        return self._extract_listings(html, region)

    def scrape(
        self,
        regions: list[str] | None = None,
        query: str = "painting",
        min_price: int | None = None,
        max_price: int | None = None,
        has_image: bool = True,
        delay: float = 1.0,
    ) -> list[Listing]:
        if regions is None:
            regions = self.CA_REGIONS

        all_listings: list[Listing] = []
        for i, region in enumerate(regions):
            listings = self._scrape_region(
                region, query, min_price=min_price, max_price=max_price, has_image=has_image,
            )
            all_listings.extend(listings)
            if i < len(regions) - 1 and delay > 0:
                time.sleep(delay)

        # Deduplicate by title
        seen: set[str] = set()
        unique: list[Listing] = []
        for listing in all_listings:
            if listing.title not in seen:
                seen.add(listing.title)
                unique.append(listing)

        # Filter junk
        before = len(unique)
        unique = [l for l in unique if self._is_likely_painting(l)]
        junk = before - len(unique)
        print(f"\nTotal: {len(unique)} paintings across {len(regions)} regions "
              f"(filtered {junk} junk from {before} unique, {len(all_listings)} raw)")
        return unique
