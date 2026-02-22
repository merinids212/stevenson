"""Craigslist US paintings scraper."""

import json
import random
import re
import time
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from .base import BaseScraper, Listing


class CraigslistScraper(BaseScraper):
    source = "cl"

    QUERY_POOL = [
        "painting", "oil painting", "original art", "artwork",
        "watercolor", "acrylic painting", "canvas art", "fine art",
    ]

    # Every US Craigslist subdomain, keyed by state
    US_REGIONS: dict[str, list[str]] = {
        "al": ["auburn", "bham", "dothan", "shoals", "gadsden", "huntsville", "mobile", "montgomery", "tuscaloosa"],
        "ak": ["anchorage", "fairbanks", "kenai", "juneau"],
        "az": ["flagstaff", "mohave", "phoenix", "prescott", "showlow", "sierravista", "tucson", "yuma"],
        "ar": ["fayar", "fortsmith", "jonesboro", "littlerock", "texarkana"],
        "ca": [
            "bakersfield", "chico", "fresno", "goldcountry", "hanford", "humboldt",
            "imperial", "inlandempire", "losangeles", "mendocino", "merced", "modesto",
            "monterey", "orangecounty", "palmsprings", "redding", "sacramento", "sandiego",
            "sfbay", "slo", "santabarbara", "santamaria", "siskiyou",
            "stockton", "susanville", "ventura", "visalia", "yubasutter",
        ],
        "co": ["boulder", "cosprings", "denver", "eastco", "fortcollins", "rockies", "pueblo", "westslope"],
        "ct": ["newlondon", "hartford", "newhaven", "nwct"],
        "de": ["delaware"],
        "dc": ["washingtondc"],
        "fl": [
            "miami", "daytona", "keys", "fortmyers", "gainesville", "cfl",
            "jacksonville", "lakeland", "lakecity", "ocala", "okaloosa", "orlando",
            "panamacity", "pensacola", "sarasota", "spacecoast", "staugustine",
            "tallahassee", "tampa", "treasure",
        ],
        "ga": ["albanyga", "athensga", "atlanta", "augusta", "brunswick", "columbusga", "macon", "nwga", "savannah", "statesboro", "valdosta"],
        "hi": ["honolulu"],
        "id": ["boise", "eastidaho", "lewiston", "twinfalls"],
        "il": ["bn", "chambana", "chicago", "decatur", "lasalle", "mattoon", "peoria", "rockford", "carbondale", "springfieldil", "quincy"],
        "in": ["bloomington", "evansville", "fortwayne", "indianapolis", "kokomo", "tippecanoe", "muncie", "richmondin", "southbend", "terrehaute"],
        "ia": ["ames", "cedarrapids", "desmoines", "dubuque", "fortdodge", "iowacity", "masoncity", "quadcities", "siouxcity", "ottumwa", "waterloo"],
        "ks": ["lawrence", "ksu", "nwks", "salina", "seks", "swks", "topeka", "wichita"],
        "ky": ["bgky", "eastky", "lexington", "louisville", "owensboro", "westky"],
        "la": ["batonrouge", "cenla", "houma", "lafayette", "lakecharles", "monroe", "neworleans", "shreveport"],
        "me": ["maine"],
        "md": ["annapolis", "baltimore", "easternshore", "frederick", "smd", "westmd"],
        "ma": ["boston", "capecod", "southcoast", "westernmass", "worcester"],
        "mi": [
            "annarbor", "battlecreek", "centralmich", "detroit", "flint", "grandrapids",
            "holland", "jxn", "kalamazoo", "lansing", "monroemi", "muskegon", "nmi",
            "porthuron", "saginaw", "swmi", "thumb", "up",
        ],
        "mn": ["bemidji", "brainerd", "duluth", "mankato", "minneapolis", "rmn", "marshall", "stcloud"],
        "ms": ["gulfport", "hattiesburg", "jackson", "meridian", "northmiss", "natchez"],
        "mo": ["columbiamo", "joplin", "kansascity", "kirksville", "loz", "semo", "springfield", "stjoseph", "stlouis"],
        "mt": ["billings", "bozeman", "butte", "greatfalls", "helena", "kalispell", "missoula", "montana"],
        "ne": ["grandisland", "lincoln", "northplatte", "omaha", "scottsbluff"],
        "nv": ["elko", "lasvegas", "reno"],
        "nh": ["nh"],
        "nj": ["cnj", "jerseyshore", "newjersey", "southjersey"],
        "nm": ["albuquerque", "clovis", "farmington", "lascruces", "roswell", "santafe"],
        "ny": [
            "albany", "binghamton", "buffalo", "catskills", "chautauqua", "elmira",
            "fingerlakes", "glensfalls", "hudsonvalley", "ithaca", "longisland",
            "newyork", "oneonta", "plattsburgh", "potsdam", "rochester", "syracuse",
            "twintiers", "utica", "watertown",
        ],
        "nc": ["asheville", "boone", "charlotte", "eastnc", "fayetteville", "greensboro", "hickory", "onslow", "outerbanks", "raleigh", "wilmington", "winstonsalem"],
        "nd": ["bismarck", "fargo", "grandforks", "nd"],
        "oh": [
            "akroncanton", "ashtabula", "athensohio", "chillicothe", "cincinnati",
            "cleveland", "columbus", "dayton", "limaohio", "mansfield", "sandusky",
            "toledo", "tuscarawas", "youngstown", "zanesville",
        ],
        "ok": ["lawton", "enid", "oklahomacity", "stillwater", "tulsa"],
        "or": ["bend", "corvallis", "eastoregon", "eugene", "klamath", "medford", "oregoncoast", "portland", "roseburg", "salem"],
        "pa": [
            "altoona", "chambersburg", "erie", "harrisburg", "lancaster", "allentown",
            "meadville", "philadelphia", "pittsburgh", "poconos", "reading", "scranton",
            "pennstate", "williamsport", "york",
        ],
        "ri": ["providence"],
        "sc": ["charleston", "columbia", "florencesc", "greenville", "hiltonhead", "myrtlebeach"],
        "sd": ["nesd", "csd", "rapidcity", "siouxfalls", "sd"],
        "tn": ["chattanooga", "clarksville", "cookeville", "jacksontn", "knoxville", "memphis", "nashville", "tricities"],
        "tx": [
            "abilene", "amarillo", "austin", "beaumont", "brownsville", "collegestation",
            "corpuschristi", "dallas", "nacogdoches", "delrio", "elpaso", "galveston",
            "houston", "killeen", "laredo", "lubbock", "mcallen", "odessa", "sanangelo",
            "sanantonio", "sanmarcos", "bigbend", "texoma", "easttexas", "victoriatx",
            "waco", "wichitafalls",
        ],
        "ut": ["logan", "ogden", "provo", "saltlakecity", "stgeorge"],
        "vt": ["vermont"],
        "va": ["charlottesville", "danville", "fredericksburg", "norfolk", "harrisonburg", "lynchburg", "blacksburg", "richmond", "roanoke", "swva", "winchester"],
        "wa": ["bellingham", "kpr", "moseslake", "olympic", "pullman", "seattle", "skagit", "spokane", "wenatchee", "yakima"],
        "wv": ["charlestonwv", "martinsburg", "huntington", "morgantown", "wheeling", "parkersburg", "swv", "wv"],
        "wi": ["appleton", "eauclaire", "greenbay", "janesville", "racine", "lacrosse", "madison", "milwaukee", "northernwi", "sheboygan", "wausau"],
        "wy": ["wyoming"],
    }

    # Reverse lookup: region -> state
    _REGION_TO_STATE: dict[str, str] = {}
    for _st, _regions in US_REGIONS.items():
        for _r in _regions:
            _REGION_TO_STATE[_r] = _st

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    JUNK_KEYWORDS = [
        # vehicles
        "sedan", "coupe", "suv", "honda", "toyota", "ford", "chevy", "bmw",
        "lexus", "audi", "nissan", "hyundai", "kia", "jeep", "dodge",
        "chrysler", "cadillac", "lincoln", "volvo", "subaru", "mazda",
        "mercedes", "benz", "scion", "truck", "car ", "trailer", "forklift",
        # construction / industrial
        "compressor", "scaffold", "scissor lift", "cnc mill", "manufacturing",
        "lathe", "grinder", "equipment", "pvc", "lumber", "plywood",
        "drywall", "concrete", "roofing", "flooring", "tile",
        "deck mat", "deck material", "scaffolding",
        "countertop", "counter top", "quartz", "granite slab", "backsplash",
        # appliances / furniture
        "cabinet", "appliance", "washer", "refrigerator", "dishwasher",
        "microwave", "stove", "mattress", "dryer", "coffee table",
        "ikea", "armchair", "recliner", "sofa", "couch", "loveseat",
        "dresser", "nightstand", "bookshelf", "bookcase",
        "office chair", "office desk", "dining table", "dining set", "bar stool",
        "futon", "bunk bed", "baby crib", "high chair", "playpen",
        # kids / toys / misc non-art
        "kids table", "toy", "lego", "playset", "trampoline",
        "book", "novel", "textbook", "ppe", "clean suit",
        "foot cover", "face shield", "golf course",
        # services (painting services, not actual paintings)
        "paint job", "painter needed", "painters needed",
        "painting service", "painting company", "painting business",
        "painting contractor", "painting crew", "paint crew",
        "painters llc", "painting llc",
        "paint sprayer", "paint roller", "paint brush set",
        "gallon of", "sherwin", "benjamin moore", "behr paint",
        "sandblasting", "leaky metal roof",
        "commercial paint", "face painting business",
        # vehicles / heavy equipment
        "articulating boom", "boom lift", "boomlift",
        "replicas starting",
        # spam / non-art
        "sculptorist", "investors team",
        "service tech",
    ]

    @classmethod
    def all_regions(cls) -> list[str]:
        """Flat list of every US Craigslist region."""
        return [r for regions in cls.US_REGIONS.values() for r in regions]

    @classmethod
    def regions_for_state(cls, state: str) -> list[str]:
        """Get regions for a two-letter state code."""
        return cls.US_REGIONS.get(state.lower(), [])

    @classmethod
    def state_for_region(cls, region: str) -> str:
        """Look up state code for a region subdomain."""
        return cls._REGION_TO_STATE.get(region, "")

    def _extract_id(self, url: str) -> str:
        m = re.search(r"/(\d+)\.html", url)
        return m.group(1) if m else ""

    def _build_search_url(self, region: str, query: str, offset: int = 0, **filters) -> str:
        base = f"https://{region}.craigslist.org/search/sss"
        params = {"query": query, "sort": "date"}
        if offset > 0:
            params["s"] = offset
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
        state = self.state_for_region(region)

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
                    state=state,
                )
            )
        return results

    ART_KEYWORDS = [
        "paint", "art", "canvas", "oil", "watercolor", "acrylic",
        "framed", "lithograph", "print", "portrait", "landscape",
        "sculpture", "drawing", "sketch", "etching", "serigraph",
        "giclée", "giclee", "pastel", "gouache", "encaustic",
        "mural", "gallery",
    ]

    def _is_likely_painting(self, listing: Listing) -> bool:
        title = listing.title.lower()
        # Price sanity — CL listings over $500K are spam/jokes
        if listing.price is not None and listing.price > 500_000:
            return False
        for kw in self.JUNK_KEYWORDS:
            if kw in title:
                return False
        if not any(kw in title for kw in self.ART_KEYWORDS):
            return False
        return True

    RESULTS_PER_PAGE = 120  # CL returns up to 120 results per page

    def _scrape_region(
        self, region: str, query: str = "painting",
        min_price: int | None = None, max_price: int | None = None,
        has_image: bool = True, max_pages: int = 1, delay: float = 1.0,
    ) -> list[Listing]:
        all_listings: list[Listing] = []
        for page in range(max_pages):
            offset = page * self.RESULTS_PER_PAGE
            url = self._build_search_url(
                region, query, offset=offset,
                min_price=min_price, max_price=max_price, has_image=has_image,
            )
            if page == 0:
                print(f"  {region}: {url}")
            else:
                print(f"  {region} p{page+1}: {url}")
            html = self._fetch_page(url)
            if not html:
                break
            page_listings = self._extract_listings(html, region)
            if not page_listings:
                break
            all_listings.extend(page_listings)
            if page < max_pages - 1 and delay > 0:
                time.sleep(delay)
        return all_listings

    def scrape(
        self,
        regions: list[str] | None = None,
        states: list[str] | None = None,
        query: str = "painting",
        queries: list[str] | None = None,
        num_queries: int | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        has_image: bool = True,
        delay: float = 1.0,
        max_pages: int = 1,
        known_ids: set[str] | None = None,
    ):
        """Yield batches of filtered, deduped listings per region."""
        if regions is not None:
            target = regions
        elif states is not None:
            target = []
            for st in states:
                target.extend(self.regions_for_state(st))
        else:
            target = self.all_regions()

        # Resolve query list
        if queries:
            query_list = queries
        elif num_queries and num_queries > 1:
            query_list = random.sample(
                self.QUERY_POOL, min(num_queries, len(self.QUERY_POOL))
            )
        else:
            query_list = [query]

        print(f"  Queries: {query_list}", flush=True)
        print(f"  Max pages per region/query: {max_pages}", flush=True)
        print(f"  Regions: {len(target)}\n", flush=True)

        if known_ids is None:
            known_ids = set()

        seen_ids: set[str] = set(known_ids)
        seen_titles: set[str] = set()

        for i, region in enumerate(target):
            region_batch: list[Listing] = []

            for q in query_list:
                listings = self._scrape_region(
                    region, q, min_price=min_price, max_price=max_price,
                    has_image=has_image, max_pages=max_pages, delay=delay,
                )

                for listing in listings:
                    if listing.id and listing.id in seen_ids:
                        continue
                    if listing.id:
                        seen_ids.add(listing.id)
                    region_batch.append(listing)

            # Dedup by title (cross-region reposts)
            deduped = []
            for listing in region_batch:
                if listing.title not in seen_titles:
                    seen_titles.add(listing.title)
                    deduped.append(listing)

            # Filter junk
            clean = [l for l in deduped if self._is_likely_painting(l)]

            if clean:
                print(f"    → {region}: {len(clean)} paintings", flush=True)
                yield clean

            if i < len(target) - 1 and delay > 0:
                time.sleep(delay)
