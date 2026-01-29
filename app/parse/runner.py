"""
Parse runner with fallback support.
Orchestrates content fetching across multiple providers with automatic fallback.
"""

import logging
from typing import Optional

from app.config import get_runtime_config
from app.parse.base import ParseResult, ParseProvider
from app.parse.providers.exa import ExaProvider
from app.parse.providers.direct import DirectProvider

logger = logging.getLogger(__name__)

# Registry of available providers
PROVIDERS: dict[str, type[ParseProvider]] = {
    "exa": ExaProvider,
    "direct": DirectProvider,
}


def get_provider(name: str) -> Optional[ParseProvider]:
    """Get a provider instance by name."""
    provider_class = PROVIDERS.get(name)
    if provider_class:
        return provider_class()
    return None


def is_parse_enabled() -> bool:
    """
    Check if any parse provider is enabled.
    Returns True if at least one provider in the configured order is enabled.
    """
    config = get_runtime_config()
    for provider_name in config.parse_providers_order:
        provider = get_provider(provider_name)
        if provider and provider.is_enabled():
            return True
    return False


async def fetch_content_with_fallback(
    urls: list[str],
    providers_order: Optional[list[str]] = None,
    min_text_chars: Optional[int] = None,
) -> dict[str, ParseResult]:
    """
    Fetch content from URLs using providers in order, with automatic fallback.

    For each URL:
    - Try the first provider
    - If it fails or text doesn't meet threshold, try the next provider
    - Continue until success or all providers exhausted

    Args:
        urls: List of URLs to fetch.
        providers_order: Order of providers to try (defaults to config).
        min_text_chars: Minimum text length threshold (defaults to config).

    Returns:
        Dict mapping URL -> ParseResult (from the first successful provider,
        or the last failed result if all providers failed).
    """
    if not urls:
        return {}

    config = get_runtime_config()

    if providers_order is None:
        providers_order = config.parse_providers_order

    if min_text_chars is None:
        min_text_chars = config.parse_min_text_chars

    # Track results for each URL
    final_results: dict[str, ParseResult] = {}
    # Track URLs that still need processing
    remaining_urls = set(urls)
    # Track failure reasons for logging
    failure_log: dict[str, list[str]] = {url: [] for url in urls}

    for provider_name in providers_order:
        if not remaining_urls:
            break

        provider = get_provider(provider_name)
        if not provider:
            logger.warning(f"Unknown provider: {provider_name}, skipping")
            continue

        if not provider.is_enabled():
            logger.debug(f"Provider {provider_name} not enabled, skipping")
            for url in remaining_urls:
                failure_log[url].append(f"{provider_name}: not enabled")
            continue

        logger.info(f"Trying provider '{provider_name}' for {len(remaining_urls)} URLs")

        # Fetch from this provider
        try:
            provider_results = await provider.fetch(list(remaining_urls))
        except Exception as e:
            logger.error(f"Provider {provider_name} failed with exception: {e}")
            for url in remaining_urls:
                failure_log[url].append(f"{provider_name}: exception - {str(e)}")
            continue

        # Process results
        urls_to_remove = set()
        for url in remaining_urls:
            result = provider_results.get(url)

            if result is None:
                failure_log[url].append(f"{provider_name}: no result returned")
                continue

            if result.success and result.meets_threshold(min_text_chars):
                # Success! Store result and mark URL as done
                final_results[url] = result
                urls_to_remove.add(url)
                logger.debug(
                    f"URL {url} succeeded with {provider_name} "
                    f"(text: {len(result.text or '')} chars)"
                )
            else:
                # Failed or below threshold
                reason = result.error or f"text below threshold ({len(result.text or '')} < {min_text_chars})"
                failure_log[url].append(f"{provider_name}: {reason}")
                # Keep the result in case all providers fail
                if url not in final_results:
                    final_results[url] = result

        remaining_urls -= urls_to_remove

    # Log failures for URLs that didn't succeed with any provider
    for url in remaining_urls:
        reasons = failure_log.get(url, [])
        logger.warning(
            f"All providers failed for URL {url}: {'; '.join(reasons)}"
        )
        # If we don't have any result stored, create a failure result
        if url not in final_results:
            final_results[url] = ParseResult(
                url=url,
                provider="none",
                success=False,
                error=f"All providers failed: {'; '.join(reasons)}",
            )

    # Summary log
    success_count = sum(1 for r in final_results.values() if r.success and r.meets_threshold(min_text_chars))
    logger.info(
        f"Parse complete: {success_count}/{len(urls)} URLs succeeded "
        f"(providers: {providers_order})"
    )

    return final_results


def truncate_text(text: Optional[str], max_chars: Optional[int] = None) -> Optional[str]:
    """
    Truncate text to max_chars, adding '...' if truncated.

    Args:
        text: Text to truncate.
        max_chars: Maximum characters (defaults to config.exa_text_max_chars for compatibility).

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
