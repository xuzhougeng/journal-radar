"""
Crossref API source implementation.
https://api.crossref.org/swagger-ui/index.html
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx
from dateutil import parser as date_parser

from app.config import get_runtime_config
from app.sources.base import BaseSource, Entry

logger = logging.getLogger(__name__)

# Crossref API base URL
CROSSREF_API_BASE = "https://api.crossref.org"


class CrossrefSource(BaseSource):
    """Fetch entries from Crossref API by ISSN."""

    @property
    def source_type(self) -> str:
        return "crossref"

    def _get_issn(self) -> str:
        """Get ISSN from config."""
        return self.config.get("issn", "")

    def _get_journal_name(self) -> str:
        """Get journal name from config (optional)."""
        return self.config.get("journal_name", "")

    def _get_rows(self) -> int:
        """Get number of rows to fetch (default 20)."""
        return self.config.get("rows", 20)

    async def fetch_entries(self) -> list[Entry]:
        """Fetch entries from Crossref API."""
        issn = self._get_issn()
        if not issn:
            logger.error(f"No ISSN configured for subscription {self.subscription_id}")
            return []

        # Normalize ISSN (remove hyphens if present, then add back)
        issn = issn.replace("-", "")
        if len(issn) == 8:
            issn = f"{issn[:4]}-{issn[4:]}"

        try:
            works = await self._fetch_works(issn)
            entries = []

            for work in works:
                entry = self._parse_work(work)
                if entry:
                    entries.append(entry)

            logger.info(f"Fetched {len(entries)} entries from Crossref for ISSN {issn}")
            return entries

        except Exception as e:
            logger.error(f"Error fetching from Crossref for ISSN {issn}: {e}")
            raise

    async def _fetch_works(self, issn: str) -> list[dict]:
        """Fetch works from Crossref API for a given ISSN."""
        config = get_runtime_config()
        
        # Use the works endpoint with ISSN filter, sorted by published date
        url = f"{CROSSREF_API_BASE}/journals/{issn}/works"

        params = {
            "rows": self._get_rows(),
            "sort": "published",
            "order": "desc",
            "select": "DOI,title,author,published,link,abstract,container-title,URL",
        }

        headers = {
            "User-Agent": config.user_agent,
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            # Crossref API has rate limiting (polite pool requires mailto in User-Agent)
            response = await client.get(url, params=params, headers=headers)

            if response.status_code == 404:
                logger.warning(f"ISSN {issn} not found in Crossref")
                return []

            response.raise_for_status()
            data = response.json()

            if "message" in data and "items" in data["message"]:
                return data["message"]["items"]

            return []

    def _parse_work(self, work: dict) -> Optional[Entry]:
        """Parse a Crossref work into a standardized Entry object."""
        # Get title (required)
        titles = work.get("title", [])
        if not titles:
            return None
        title = titles[0] if isinstance(titles, list) else str(titles)
        title = title.strip()

        if not title:
            return None

        # Get DOI (required for Crossref entries)
        doi = work.get("DOI", "")
        if not doi:
            return None

        # Generate link from DOI
        link = work.get("URL", f"https://doi.org/{doi}")

        # Get published date
        published_at = self._parse_date(work)

        # Get authors
        authors = self._extract_authors(work)

        # Get abstract
        abstract = work.get("abstract")
        if abstract:
            # Clean up JATS XML tags often present in Crossref abstracts
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()
            if len(abstract) > 1000:
                abstract = abstract[:997] + "..."

        # Get journal name
        container_titles = work.get("container-title", [])
        journal_name = container_titles[0] if container_titles else self._get_journal_name()

        return Entry(
            title=title,
            link=link,
            source_id=self.subscription_id,
            published_at=published_at,
            doi=doi,
            authors=authors,
            abstract=abstract,
            journal_name=journal_name,
            raw_data=work,
        )

    def _parse_date(self, work: dict) -> Optional[datetime]:
        """Parse publication date from Crossref work."""
        # Try different date fields in order of preference
        date_fields = ["published", "published-print", "published-online", "created"]

        for field in date_fields:
            if field in work and work[field]:
                date_parts = work[field].get("date-parts", [[]])
                if date_parts and date_parts[0]:
                    parts = date_parts[0]
                    try:
                        year = parts[0] if len(parts) > 0 else 1970
                        month = parts[1] if len(parts) > 1 else 1
                        day = parts[2] if len(parts) > 2 else 1
                        return datetime(year, month, day)
                    except (ValueError, TypeError):
                        continue

        return None

    def _extract_authors(self, work: dict) -> Optional[str]:
        """Extract author names from Crossref work."""
        authors = work.get("author", [])
        if not authors:
            return None

        names = []
        for author in authors[:10]:  # Limit to first 10 authors
            given = author.get("given", "")
            family = author.get("family", "")
            if family:
                if given:
                    names.append(f"{given} {family}")
                else:
                    names.append(family)
            elif author.get("name"):
                names.append(author["name"])

        if names:
            result = ", ".join(names)
            if len(authors) > 10:
                result += f" et al. ({len(authors)} authors)"
            return result

        return None


async def search_journals(query: str, rows: int = 10) -> list[dict]:
    """
    Search for journals in Crossref.
    Useful for finding ISSNs when you know the journal name.

    Args:
        query: Journal name or partial name to search for
        rows: Number of results to return

    Returns:
        List of journal info dictionaries with title, ISSNs, publisher
    """
    config = get_runtime_config()
    url = f"{CROSSREF_API_BASE}/journals"

    params = {
        "query": query,
        "rows": rows,
    }

    headers = {
        "User-Agent": config.user_agent,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=config.request_timeout) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        journals = []
        if "message" in data and "items" in data["message"]:
            for item in data["message"]["items"]:
                journals.append({
                    "title": item.get("title", ""),
                    "issn": item.get("ISSN", []),
                    "publisher": item.get("publisher", ""),
                    "subjects": item.get("subjects", []),
                })

        return journals
