"""
Exa AI parse provider.
Wraps existing exa_ai module for the parse abstraction layer.
"""

import logging
from typing import Optional

from app.config import get_runtime_config
from app.parse.base import ParseProvider, ParseResult

logger = logging.getLogger(__name__)


class ExaProvider(ParseProvider):
    """Parse provider using Exa AI content extraction API."""

    name = "exa"

    def is_enabled(self) -> bool:
        """Check if Exa API key is configured."""
        config = get_runtime_config()
        return bool(config.exa_api_key)

    async def fetch(self, urls: list[str]) -> dict[str, ParseResult]:
        """
        Fetch content via Exa AI.

        Args:
            urls: List of URLs to fetch.

        Returns:
            Dict mapping URL -> ParseResult.
        """
        from app.exa_ai import fetch_contents

        if not urls:
            return {}

        if not self.is_enabled():
            logger.debug("Exa provider not enabled, skipping")
            return {
                url: ParseResult(
                    url=url,
                    provider=self.name,
                    success=False,
                    error="Exa API key not configured",
                )
                for url in urls
            }

        results: dict[str, ParseResult] = {}

        try:
            exa_response = await fetch_contents(urls)

            if not exa_response:
                logger.warning("Exa API returned no response")
                return {
                    url: ParseResult(
                        url=url,
                        provider=self.name,
                        success=False,
                        error="Exa API returned no response",
                    )
                    for url in urls
                }

            # Build URL -> Exa result mapping
            exa_results_map: dict[str, any] = {}
            for r in exa_response.results:
                if r.id:
                    exa_results_map[r.id] = r
                if r.url and r.url != r.id:
                    exa_results_map[r.url] = r

            # Build URL -> error mapping
            exa_errors_map: dict[str, any] = {}
            for e in exa_response.errors:
                if e.id:
                    exa_errors_map[e.id] = e

            # Convert to ParseResult for each URL
            for url in urls:
                if url in exa_errors_map:
                    error = exa_errors_map[url]
                    results[url] = ParseResult(
                        url=url,
                        provider=self.name,
                        success=False,
                        error=f"Exa error: {error.error_tag or 'unknown'}",
                        status="error",
                    )
                elif url in exa_results_map:
                    r = exa_results_map[url]
                    results[url] = ParseResult(
                        url=url,
                        provider=self.name,
                        success=True,
                        text=r.text,
                        title=r.title,
                        author=r.author,
                        final_url=r.url,
                        raw_path=exa_response.raw_path,
                        status=r.status,
                        metadata={
                            "request_id": exa_response.request_id,
                            "cost_total": exa_response.cost_total,
                            "cost_text": exa_response.cost_text,
                            "search_time_ms": exa_response.search_time_ms,
                        },
                    )
                else:
                    results[url] = ParseResult(
                        url=url,
                        provider=self.name,
                        success=False,
                        error="URL not found in Exa response",
                    )

            return results

        except Exception as e:
            logger.error(f"Exa provider error: {e}")
            return {
                url: ParseResult(
                    url=url,
                    provider=self.name,
                    success=False,
                    error=f"Exa provider exception: {str(e)}",
                )
                for url in urls
            }
