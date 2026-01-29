"""
Base classes and data structures for parse providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParseResult:
    """Result from a parse provider for a single URL."""

    url: str  # Original requested URL
    provider: str  # Provider name (e.g., 'exa', 'direct')
    success: bool  # Whether extraction was successful
    
    # Content fields (populated on success)
    text: Optional[str] = None  # Extracted text content
    title: Optional[str] = None  # Page title
    author: Optional[str] = None  # Author if available
    final_url: Optional[str] = None  # Final URL after redirects
    
    # Metadata
    raw_path: Optional[str] = None  # Relative path to raw data file
    status: Optional[str] = None  # Provider-specific status
    error: Optional[str] = None  # Error message if failed
    
    # Provider-specific metadata (e.g., cost, timing)
    metadata: dict = field(default_factory=dict)
    
    def meets_threshold(self, min_chars: int) -> bool:
        """Check if extracted text meets minimum character threshold."""
        if not self.success or not self.text:
            return False
        return len(self.text) >= min_chars


@dataclass
class ParseError:
    """Error info for a failed URL."""

    url: str
    provider: str
    error: str
    error_tag: Optional[str] = None  # Provider-specific error code


class ParseProvider(ABC):
    """Abstract base class for parse providers."""

    name: str = "base"

    @abstractmethod
    async def fetch(self, urls: list[str]) -> dict[str, ParseResult]:
        """
        Fetch and parse content from URLs.

        Args:
            urls: List of URLs to fetch.

        Returns:
            Dict mapping URL -> ParseResult.
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if this provider is configured and enabled."""
        pass
