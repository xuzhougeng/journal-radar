"""
Direct fetch parse provider.
Fetches web pages directly and extracts readable content using readability-lxml.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from app.config import get_runtime_config, StaticConfig
from app.parse.base import ParseProvider, ParseResult

logger = logging.getLogger(__name__)


def extract_readable_content(html: str, url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract readable content from HTML using readability-lxml.

    Args:
        html: Raw HTML content.
        url: Original URL (for resolving relative links).

    Returns:
        Tuple of (text_content, title).
    """
    try:
        from readability import Document

        doc = Document(html, url=url)
        title = doc.title()
        
        # Get the cleaned HTML content
        content_html = doc.summary()
        
        # Strip HTML tags to get plain text
        from html.parser import HTMLParser
        
        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self.in_script = False
                self.in_style = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self.in_script = tag == "script"
                    self.in_style = tag == "style"
                elif tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
                    self.text_parts.append("\n")

            def handle_endtag(self, tag):
                if tag == "script":
                    self.in_script = False
                elif tag == "style":
                    self.in_style = False
                elif tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
                    self.text_parts.append("\n")

            def handle_data(self, data):
                if not self.in_script and not self.in_style:
                    self.text_parts.append(data)

        extractor = TextExtractor()
        extractor.feed(content_html)
        
        # Join and clean up whitespace
        text = "".join(extractor.text_parts)
        # Normalize whitespace: collapse multiple newlines, strip lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)
        
        return text, title

    except ImportError:
        logger.error("readability-lxml not installed. Install with: pip install readability-lxml")
        return None, None
    except Exception as e:
        logger.error(f"Error extracting readable content: {e}")
        return None, None


class DirectProvider(ParseProvider):
    """Parse provider that fetches pages directly via HTTP."""

    name = "direct"

    def is_enabled(self) -> bool:
        """Direct provider is always enabled (no API key needed)."""
        return True

    async def fetch(self, urls: list[str]) -> dict[str, ParseResult]:
        """
        Fetch content directly via HTTP and extract readable text.

        Args:
            urls: List of URLs to fetch.

        Returns:
            Dict mapping URL -> ParseResult.
        """
        if not urls:
            return {}

        config = get_runtime_config()
        results: dict[str, ParseResult] = {}

        # Ensure data directory exists
        data_dir = StaticConfig.get_parse_data_dir("direct")
        data_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(
            timeout=config.request_timeout,
            follow_redirects=True,
            headers={"User-Agent": config.user_agent},
        ) as client:
            for url in urls:
                result = await self._fetch_single(client, url, data_dir)
                results[url] = result

        return results

    async def _fetch_single(
        self, client: httpx.AsyncClient, url: str, data_dir: Path
    ) -> ParseResult:
        """Fetch and parse a single URL."""
        try:
            response = await client.get(url)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower() and "application/xhtml" not in content_type.lower():
                return ParseResult(
                    url=url,
                    provider=self.name,
                    success=False,
                    error=f"Not HTML content: {content_type}",
                )

            html = response.text
            final_url = str(response.url)

            # Extract readable content
            text, title = extract_readable_content(html, final_url)

            if not text:
                return ParseResult(
                    url=url,
                    provider=self.name,
                    success=False,
                    error="Failed to extract readable content",
                    final_url=final_url,
                )

            # Save raw HTML to disk
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            # Create a safe filename from URL
            safe_url = url.replace("://", "_").replace("/", "_").replace("?", "_")[:50]
            filename = f"{timestamp}_{safe_url}.json"
            raw_path = data_dir / filename

            raw_data = {
                "url": url,
                "final_url": final_url,
                "title": title,
                "text": text,
                "html": html,
                "fetched_at": datetime.utcnow().isoformat(),
                "status_code": response.status_code,
                "content_type": content_type,
            }

            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)

            relative_path = str(Path("data/parse/direct") / filename)

            logger.info(f"Direct fetch saved to {raw_path}")

            return ParseResult(
                url=url,
                provider=self.name,
                success=True,
                text=text,
                title=title,
                final_url=final_url,
                raw_path=relative_path,
                status="success",
                metadata={
                    "status_code": response.status_code,
                    "content_length": len(html),
                    "text_length": len(text),
                },
            )

        except httpx.HTTPStatusError as e:
            return ParseResult(
                url=url,
                provider=self.name,
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                status="http_error",
            )
        except httpx.RequestError as e:
            return ParseResult(
                url=url,
                provider=self.name,
                success=False,
                error=f"Request error: {str(e)}",
                status="request_error",
            )
        except Exception as e:
            logger.error(f"Direct fetch error for {url}: {e}")
            return ParseResult(
                url=url,
                provider=self.name,
                success=False,
                error=f"Unexpected error: {str(e)}",
                status="error",
            )
