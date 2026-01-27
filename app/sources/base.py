"""
Base class and common data structures for journal sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib


@dataclass
class Entry:
    """Standardized entry from any journal source."""

    title: str
    link: str
    source_id: int  # Reference to subscription in database
    published_at: Optional[datetime] = None
    doi: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    journal_name: Optional[str] = None
    raw_data: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """
        Generate a unique fingerprint for deduplication.
        Priority: DOI > (normalized_title + published_date + link)
        """
        if self.doi:
            # Normalize DOI: lowercase, strip whitespace
            normalized_doi = self.doi.lower().strip()
            return hashlib.sha256(f"doi:{normalized_doi}".encode()).hexdigest()

        # Fallback: combine title, date, and link
        normalized_title = self.title.lower().strip()
        date_str = self.published_at.isoformat() if self.published_at else ""
        combined = f"{normalized_title}|{date_str}|{self.link}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Convert entry to dictionary for serialization."""
        return {
            "title": self.title,
            "link": self.link,
            "source_id": self.source_id,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "doi": self.doi,
            "authors": self.authors,
            "abstract": self.abstract,
            "journal_name": self.journal_name,
            "fingerprint": self.fingerprint,
        }


class BaseSource(ABC):
    """
    Abstract base class for journal sources.
    All source implementations (RSS, Crossref, TOC scraper, etc.) should inherit from this.
    """

    def __init__(self, subscription_id: int, config: dict):
        """
        Initialize a source.

        Args:
            subscription_id: Database ID of the subscription
            config: Source-specific configuration (e.g., feed URL, ISSN)
        """
        self.subscription_id = subscription_id
        self.config = config

    @abstractmethod
    async def fetch_entries(self) -> list[Entry]:
        """
        Fetch entries from the source.

        Returns:
            List of standardized Entry objects
        """
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type identifier (e.g., 'rss', 'crossref')."""
        pass
