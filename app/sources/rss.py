"""
RSS/Atom feed source implementation.
"""

import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import feedparser
import httpx
from dateutil import parser as date_parser

from app.config import get_runtime_config
from app.sources.base import BaseSource, Entry

logger = logging.getLogger(__name__)


class RSSSource(BaseSource):
    """Fetch entries from RSS/Atom feeds."""

    @property
    def source_type(self) -> str:
        return "rss"

    def _get_feed_url(self) -> str:
        """Get feed URL from config."""
        return self.config.get("feed_url", "")

    async def fetch_entries(self) -> list[Entry]:
        """Fetch and parse entries from the RSS feed."""
        feed_url = self._get_feed_url()
        if not feed_url:
            logger.error(f"No feed URL configured for subscription {self.subscription_id}")
            return []

        config = get_runtime_config()

        try:
            # Fetch the feed content
            async with httpx.AsyncClient(timeout=config.request_timeout) as client:
                response = await client.get(
                    feed_url,
                    headers={"User-Agent": config.user_agent},
                    follow_redirects=True,
                )
                response.raise_for_status()
                content = response.text

            # Parse the feed
            feed = feedparser.parse(content)

            if feed.bozo and feed.bozo_exception:
                logger.warning(
                    f"Feed parse warning for {feed_url}: {feed.bozo_exception}"
                )

            entries = []
            for item in feed.entries:
                entry = self._parse_entry(item, feed.feed, feed_url)
                if entry:
                    entries.append(entry)

            logger.info(f"Fetched {len(entries)} entries from {feed_url}")
            return entries

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {feed_url}: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {feed_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing feed {feed_url}: {e}")
            raise

    def _parse_entry(self, item: dict, feed_info: dict, feed_url: str) -> Optional[Entry]:
        """Parse a single feed entry into a standardized Entry object."""
        # Get title (required)
        title = item.get("title", "").strip()
        if not title:
            return None

        # Get link (required)
        link = item.get("link", "")
        if not link and item.get("links"):
            # Try to get the first link
            link = item.links[0].get("href", "")
        if not link:
            return None

        # Get DOI if available
        doi = self._extract_doi(item)

        if self._should_skip_item(feed_url=feed_url, title=title, link=link, doi=doi):
            return None

        # Get published date
        published_at = self._parse_date(item)

        # Get authors
        authors = self._extract_authors(item)

        # Get abstract/summary
        abstract = None
        if item.get("summary"):
            abstract = item.summary
        elif item.get("description"):
            abstract = item.description

        # Clean up abstract (remove HTML tags for display)
        if abstract:
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()
            # Limit length
            if len(abstract) > 1000:
                abstract = abstract[:997] + "..."

        # Get journal name from feed title if not in entry
        journal_name = feed_info.get("title", "")

        return Entry(
            title=title,
            link=link,
            source_id=self.subscription_id,
            published_at=published_at,
            doi=doi,
            authors=authors,
            abstract=abstract,
            journal_name=journal_name,
            raw_data=dict(item),
        )

    @staticmethod
    def _is_nature_feed(feed_url: str) -> bool:
        """
        Detect Nature's mixed-content RSS feeds (e.g. nature.rss / current RSS),
        which include both research articles and editorial/news items.
        """
        try:
            parsed = urlparse(feed_url)
        except Exception:
            return False

        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        return ("nature.com" in host) and (path.endswith("/nature.rss") or "/nature/rss" in path)

    def _should_skip_item(self, feed_url: str, title: str, link: str, doi: Optional[str]) -> bool:
        """Apply per-feed heuristics to skip irrelevant items."""
        if not self._is_nature_feed(feed_url):
            return False

        title_l = title.lower()

        # Keep the feed focused on research articles; drop non-research meta items.
        exclude_title_keywords = [
            "correction",
            "expression of concern",
            "obituary",
            "daily briefing",
            "briefing chat",
            "books in brief",
        ]
        if any(k in title_l for k in exclude_title_keywords):
            return True

        return False

    def _extract_doi(self, item: dict) -> Optional[str]:
        """Extract DOI from feed entry if available."""
        # Check common DOI fields
        if item.get("prism_doi"):
            return item.prism_doi
        if item.get("dc_identifier"):
            identifier = item.dc_identifier
            if identifier.startswith("doi:"):
                return identifier[4:]
            if identifier.startswith("10."):
                return identifier

        # Check for DOI in links
        for link in item.get("links", []):
            href = link.get("href", "")
            if "doi.org/" in href:
                # Extract DOI from URL
                parts = href.split("doi.org/")
                if len(parts) > 1:
                    return parts[1]

        # Check for DOI in ID
        entry_id = item.get("id", "")
        if "doi.org/" in entry_id:
            parts = entry_id.split("doi.org/")
            if len(parts) > 1:
                return parts[1]

        return None

    def _parse_date(self, item: dict) -> Optional[datetime]:
        """Parse publication date from feed entry."""
        date_fields = [
            "published_parsed",
            "updated_parsed",
            "created_parsed",
        ]

        for field in date_fields:
            if item.get(field):
                try:
                    time_struct = item[field]
                    return datetime(*time_struct[:6])
                except (TypeError, ValueError):
                    continue

        # Try string date fields
        string_fields = ["published", "updated", "created", "dc_date"]
        for field in string_fields:
            if item.get(field):
                try:
                    return date_parser.parse(item[field])
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_authors(self, item: dict) -> Optional[str]:
        """Extract author names from feed entry."""
        # Check for author field
        if item.get("author"):
            return item.author

        # Check for authors list
        if item.get("authors"):
            names = []
            for author in item.authors:
                if isinstance(author, dict):
                    name = author.get("name", "")
                else:
                    name = str(author)
                if name:
                    names.append(name)
            if names:
                return ", ".join(names)

        # Check for dc:creator
        if item.get("dc_creator"):
            return item.dc_creator

        return None
