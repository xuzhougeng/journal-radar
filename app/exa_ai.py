"""
Exa AI content extraction client.
https://docs.exa.ai/reference/contents
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from app.config import get_runtime_config, StaticConfig

logger = logging.getLogger(__name__)

# Exa API endpoint
EXA_CONTENTS_URL = "https://api.exa.ai/contents"


@dataclass
class ExaResult:
    """Parsed result from Exa contents API for a single URL."""

    id: str  # Original requested URL
    url: str  # Actual URL returned by Exa
    title: Optional[str]
    author: Optional[str]
    text: Optional[str]
    status: str  # 'success', 'failed', etc.
    image: Optional[str] = None
    favicon: Optional[str] = None


@dataclass
class ExaError:
    """Error info from Exa statuses."""

    id: str  # URL that failed
    status: str  # 'error'
    error_tag: Optional[str] = None  # e.g. 'CRAWL_LIVECRAWL_TIMEOUT'


@dataclass
class ExaResponse:
    """Full response from Exa contents API."""

    request_id: str
    results: list[ExaResult]
    errors: list[ExaError]  # Failed URLs with error info
    cost_total: float
    cost_text: float
    search_time_ms: float
    raw_path: str  # Relative path to saved raw JSON


def is_exa_enabled() -> bool:
    """Check if Exa AI is configured and enabled."""
    config = get_runtime_config()
    return bool(config.exa_api_key)


async def fetch_contents(urls: list[str]) -> Optional[ExaResponse]:
    """
    Fetch web page contents via Exa AI contents API.

    Args:
        urls: List of URLs to extract content from.

    Returns:
        ExaResponse with results and metadata, or None if failed.
    """
    if not urls:
        return None

    config = get_runtime_config()

    if not config.exa_api_key:
        logger.warning("Exa AI API key not configured, skipping content extraction")
        return None

    # Ensure the exa data directory exists
    StaticConfig.ensure_exa_data_dir()

    try:
        async with httpx.AsyncClient(timeout=config.request_timeout * 2) as client:
            response = await client.post(
                EXA_CONTENTS_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": config.exa_api_key,
                },
                json={
                    "ids": urls,
                    "text": True,
                    # Use livecrawl fallback: try live crawl first, fall back to cache if it fails/times out
                    # This helps avoid CRAWL_LIVECRAWL_TIMEOUT errors while still getting fresh content
                    "livecrawl": "fallback",
                    "livecrawlTimeout": 15000,  # 15 seconds timeout for live crawl
                },
            )
            response.raise_for_status()
            data = response.json()

        # Save raw response to disk
        request_id = data.get("requestId", "unknown")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{request_id}_{timestamp}.json"
        raw_path = StaticConfig.get_exa_data_dir() / filename

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Relative path for database storage
        relative_raw_path = str(Path("data/exa") / filename)

        logger.info(f"Exa response saved to {raw_path}")

        # Parse results
        results_raw = data.get("results", [])
        statuses_raw = data.get("statuses", [])

        # Build status lookup by id
        status_map = {s.get("id"): s.get("status", "unknown") for s in statuses_raw}

        results = []
        for r in results_raw:
            result_id = r.get("id", "")
            results.append(
                ExaResult(
                    id=result_id,
                    url=r.get("url", result_id),
                    title=r.get("title"),
                    author=r.get("author"),
                    text=r.get("text"),
                    status=status_map.get(result_id, "unknown"),
                    image=r.get("image"),
                    favicon=r.get("favicon"),
                )
            )

        # Parse errors from statuses
        errors = []
        for s in statuses_raw:
            if s.get("status") == "error":
                error_info = s.get("error", {})
                errors.append(
                    ExaError(
                        id=s.get("id", ""),
                        status="error",
                        error_tag=error_info.get("tag") if isinstance(error_info, dict) else None,
                    )
                )

        # Parse costs
        cost_dollars = data.get("costDollars", {})
        cost_total = cost_dollars.get("total", 0.0)
        cost_contents = cost_dollars.get("contents", {})
        cost_text = cost_contents.get("text", 0.0) if isinstance(cost_contents, dict) else 0.0
        search_time_ms = data.get("searchTime", 0.0)

        if errors:
            error_tags = [e.error_tag or "unknown" for e in errors]
            logger.warning(f"Exa errors for {len(errors)} URLs: {error_tags}")

        logger.info(
            f"Exa fetched {len(results)} results, {len(errors)} errors, cost: ${cost_total:.4f}, time: {search_time_ms:.0f}ms"
        )

        return ExaResponse(
            request_id=request_id,
            results=results,
            errors=errors,
            cost_total=cost_total,
            cost_text=cost_text,
            search_time_ms=search_time_ms,
            raw_path=relative_raw_path,
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"Exa API HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Exa API request error: {e}")
        return None
    except Exception as e:
        logger.error(f"Exa API unexpected error: {e}")
        return None


def truncate_text(text: Optional[str], max_chars: Optional[int] = None) -> Optional[str]:
    """
    Truncate text to max_chars, adding '...' if truncated.

    Args:
        text: Text to truncate.
        max_chars: Maximum characters (defaults to config.exa_text_max_chars).

    Returns:
        Truncated text or original if within limit.
    """
    if text is None:
        return None

    if max_chars is None:
        config = get_runtime_config()
        max_chars = config.exa_text_max_chars

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3] + "..."
