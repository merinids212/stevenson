"""eBay US paintings scraper (Playwright-based).

Designed for incremental accumulation — pass known_ids to skip
items you already have. Queries and sort orders are randomized
each run to explore different slices of eBay's inventory.
"""

import random
import re
import time
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from .base import BaseScraper, Listing

# Big pool of queries — each run picks a random subset
QUERY_POOL = [
    "oil painting",
    "original painting",
    "fine art painting",
    "abstract painting",
    "acrylic painting",
    "watercolor painting",
    "landscape painting",
    "portrait painting",
    "still life painting",
    "modern art painting",
    "impressionist painting",
    "contemporary painting",
    "vintage painting",
    "signed painting",
    "framed painting canvas",
    "expressionist painting",
    "seascape painting",
    "floral painting original",
    "figurative painting",
    "plein air painting",
]

# eBay sort options — rotated for variety
SORT_OPTIONS = [
    ("12", "Best Match"),
    ("10", "Newly Listed"),
    ("15", "Price: Low to High"),
    ("16", "Price: High to Low"),
]


class EbayScraper(BaseScraper):
    source = "eb"

    JUNK_KEYWORDS = [
        "print", "reproduction", "poster", "giclee", "lithograph",
        "copy", "reprint", "sticker", "decal", "wallpaper",
        "paint by number", "diamond painting", "cross stitch",
        "book", "magazine", "calendar", "greeting card",
        "painting company", "painting business", "painting service",
        "paint sprayer", "paint roller",
    ]

    def _build_search_url(self, query: str, page: int = 1,
                          sort: str = "12",
                          min_price: int | None = None,
                          max_price: int | None = None) -> str:
        params = {
            "_nkw": query,
            "_sacat": "551",       # Paintings category
            "LH_BIN": "1",        # Buy It Now only
            "LH_PrefLoc": "1",    # US-located items
            "_ipg": "240",        # max items per page
            "_pgn": str(page),
            "_sop": sort,          # sort order
        }
        if min_price is not None:
            params["_udlo"] = str(min_price)
        if max_price is not None:
            params["_udhi"] = str(max_price)
        return f"https://www.ebay.com/sch/i.html?{urlencode(params)}"

    def _extract_item_id(self, url: str) -> str:
        m = re.search(r"/itm/(\d+)", url)
        return m.group(1) if m else ""

    def _upscale_image(self, img_url: str) -> str:
        return re.sub(r"s-l\d+", "s-l1600", img_url)

    def _parse_listings(self, html: str) -> list[Listing]:
        soup = BeautifulSoup(html, "html.parser")
        results = []

        for card in soup.select("div.su-card-container"):
            # Title
            title_el = card.select_one("div.s-card__title")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            title = re.sub(r"Opens in a new window or tab$", "", title).strip()
            if title.lower() == "shop on ebay":
                continue

            # URL + item ID
            link_el = card.select_one("div.su-card-container__header a[href*='/itm/']")
            if not link_el:
                continue
            url = link_el.get("href", "")
            item_id = self._extract_item_id(url)
            if not item_id:
                continue
            url = re.sub(r"\?.*", "", url)

            # Price — first attribute row with a $ sign
            price = None
            for row in card.select("div.s-card__attribute-row"):
                text = row.get_text(strip=True)
                m = re.match(r"^\$?([\d,]+\.?\d*)", text)
                if m:
                    price = float(m.group(1).replace(",", ""))
                    break

            # Image
            images = []
            img_el = card.select_one("img")
            if img_el:
                img_src = img_el.get("src") or img_el.get("data-src") or ""
                if img_src and "gif" not in img_src.lower() and "ebaystatic" not in img_src:
                    images = [self._upscale_image(img_src)]

            # Location
            location = ""
            for row in card.select("div.s-card__attribute-row"):
                text = row.get_text(strip=True)
                if "Located in" in text:
                    location = text.removeprefix("Located in ")
                    break

            results.append(
                Listing(
                    id=f"eb:{item_id}",
                    source="eb",
                    title=title,
                    price=price,
                    url=url,
                    location=location,
                    latitude=None,
                    longitude=None,
                    images=images,
                )
            )

        return results

    def _is_likely_painting(self, listing: Listing) -> bool:
        title = listing.title.lower()
        for kw in self.JUNK_KEYWORDS:
            if kw in title:
                return False
        return True

    def scrape(
        self,
        queries: list[str] | None = None,
        num_queries: int = 4,
        min_price: int | None = None,
        max_price: int | None = None,
        max_pages: int = 5,
        delay: float = 2.0,
        known_ids: set[str] | None = None,
        **kwargs,
    ):
        """Yield batches of filtered, deduped listings per query."""
        # Pick queries: explicit list, or random subset from pool
        if queries is not None:
            run_queries = list(queries)
        else:
            run_queries = random.sample(QUERY_POOL, min(num_queries, len(QUERY_POOL)))
        random.shuffle(run_queries)

        # Pick a random sort order for this run
        sort_code, sort_name = random.choice(SORT_OPTIONS)
        print(f"  Sort: {sort_name}", flush=True)
        print(f"  Queries: {run_queries}", flush=True)

        if known_ids is None:
            known_ids = set()

        seen_ids: set[str] = set(known_ids)

        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )
            page = ctx.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            for query in run_queries:
                # Random page offset — don't always start at 1
                start_page = random.randint(1, 3)
                end_page = start_page + max_pages - 1

                print(f'\n  Query: "{query}" (pages {start_page}-{end_page})', flush=True)
                query_batch: list[Listing] = []
                blocked = False

                for pg in range(start_page, end_page + 1):
                    url = self._build_search_url(
                        query, pg, sort_code, min_price, max_price,
                    )
                    print(f"    Page {pg}: {url}", flush=True)

                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        # Humanlike wait — vary between 2-4s
                        page.wait_for_timeout(random.randint(2000, 4000))
                    except Exception as e:
                        print(f"    Failed to load: {e}", flush=True)
                        break

                    if "Pardon" in page.title():
                        print("    Blocked by eBay — stopping.", flush=True)
                        blocked = True
                        break

                    html = page.content()
                    page_listings = self._parse_listings(html)
                    if not page_listings:
                        print(f"    No results on page {pg}, stopping query.", flush=True)
                        break

                    new = 0
                    skipped = 0
                    for listing in page_listings:
                        if listing.id in seen_ids:
                            skipped += 1
                            continue
                        seen_ids.add(listing.id)
                        query_batch.append(listing)
                        new += 1
                    print(f"    Got {len(page_listings)} items, {new} new, {skipped} dupes", flush=True)

                    # Early bail: if >75% of page is dupes, move to next query
                    if len(page_listings) > 0 and skipped / len(page_listings) > 0.75:
                        print(f"    High dupe rate ({skipped}/{len(page_listings)}) — next query", flush=True)
                        break

                    # Humanlike delay between pages — vary it
                    if pg < end_page:
                        time.sleep(delay + random.uniform(-0.5, 1.5))

                # Filter junk and yield this query's batch
                clean = [l for l in query_batch if self._is_likely_painting(l)]
                if clean:
                    print(f"    → \"{query}\": {len(clean)} paintings", flush=True)
                    yield clean

                if blocked:
                    browser.close()
                    return

                # Delay between queries — longer, with variance
                if query != run_queries[-1]:
                    time.sleep(delay + random.uniform(0, 2.0))

            browser.close()
