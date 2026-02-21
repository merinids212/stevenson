"""Base scraper interface for painting sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Listing:
    """A scraped painting listing."""
    id: str
    source: str
    title: str
    price: float | None
    url: str
    location: str
    latitude: float | None
    longitude: float | None
    images: list[str] = field(default_factory=list)
    posted: str = ""
    region: str = ""
    state: str = ""


class BaseScraper(ABC):
    """Base class for painting scrapers."""
    source: str = ""

    @abstractmethod
    def scrape(self, **kwargs) -> list[Listing]:
        """Scrape listings and return a list of Listing objects."""
        ...
