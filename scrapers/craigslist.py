"""Craigslist US paintings scraper."""

import json
import re
import time
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from .base import BaseScraper, Listing


class CraigslistScraper(BaseScraper):
    source = "cl"

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
            "sfbay", "slo", "santabarbara", "santacruz", "santamaria", "siskiyou",
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
        states: list[str] | None = None,
        query: str = "painting",
        min_price: int | None = None,
        max_price: int | None = None,
        has_image: bool = True,
        delay: float = 1.0,
    ) -> list[Listing]:
        if regions is not None:
            target = regions
        elif states is not None:
            target = []
            for st in states:
                target.extend(self.regions_for_state(st))
        else:
            target = self.all_regions()

        all_listings: list[Listing] = []
        for i, region in enumerate(target):
            listings = self._scrape_region(
                region, query, min_price=min_price, max_price=max_price, has_image=has_image,
            )
            all_listings.extend(listings)
            if i < len(target) - 1 and delay > 0:
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

        state_count = len({l.state for l in unique if l.state})
        print(f"\nTotal: {len(unique)} paintings across {len(target)} regions "
              f"({state_count} states, filtered {junk} junk from {before} unique, "
              f"{len(all_listings)} raw)")
        return unique
