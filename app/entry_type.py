"""
Entry type classification utilities.
Handles type inference, priority resolution, and database operations for EntryType.
"""

import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Valid article types (same as VALID_SITE_TYPES in llm_struct.py)
VALID_ARTICLE_TYPES = {
    "paper",       # Academic paper / preprint
    "journal",     # Journal article page
    "news",        # News article
    "blog",        # Blog post
    "docs",        # Documentation / manual
    "repository",  # Code repository (GitHub, GitLab, etc.)
    "forum",       # Forum / discussion thread
    "product",     # Product page
    "dataset",     # Dataset description
    "other",       # None of the above
}


def resolve_effective_type(
    user_type: Optional[str],
    llm_type: Optional[str],
    parse_type: Optional[str],
) -> str:
    """
    Resolve the effective (final) type based on priority: user > llm > parse.
    
    Args:
        user_type: User-overridden type (highest priority).
        llm_type: LLM-inferred type.
        parse_type: Parse-inferred type (lowest priority).
    
    Returns:
        The effective type string, or "other" if none are set.
    """
    # Priority: user > llm > parse
    for t in [user_type, llm_type, parse_type]:
        if t and t in VALID_ARTICLE_TYPES:
            return t
    return "other"


def infer_parse_type(
    url: Optional[str],
    title: Optional[str],
    text: Optional[str],
) -> tuple[str, str]:
    """
    Infer article type from URL, title, and text using heuristic rules.
    
    Args:
        url: The page URL.
        title: The page title.
        text: The page text content.
    
    Returns:
        Tuple of (type, reason) where type is one of VALID_ARTICLE_TYPES.
    """
    url = url or ""
    title = (title or "").lower()
    text = (text or "").lower()
    
    # Parse URL
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
    except Exception:
        domain = ""
        path = ""
    
    # Rule 1: Code repository by domain
    repo_domains = ["github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "sr.ht"]
    for repo_domain in repo_domains:
        if repo_domain in domain:
            return "repository", f"URL domain contains '{repo_domain}'"
    
    # Rule 2: Documentation by URL path or domain
    docs_patterns = ["/docs", "/documentation", "readthedocs", "/wiki", "/manual", "/guide"]
    for pattern in docs_patterns:
        if pattern in path or pattern in domain:
            return "docs", f"URL contains '{pattern}'"
    
    # Rule 3: Dataset repositories
    dataset_domains = ["kaggle.com", "huggingface.co/datasets", "zenodo.org", "figshare.com", "dryad"]
    for ds_domain in dataset_domains:
        if ds_domain in domain or ds_domain in path:
            return "dataset", f"URL indicates dataset repository '{ds_domain}'"
    
    # Rule 4: Forum patterns
    forum_patterns = ["forum", "discuss", "community", "stackoverflow", "stackexchange", "reddit.com"]
    for pattern in forum_patterns:
        if pattern in domain or pattern in path:
            return "forum", f"URL indicates forum/discussion '{pattern}'"
    
    # Rule 5: Blog patterns
    blog_patterns = ["blog", "medium.com", "dev.to", "substack.com", "hashnode"]
    for pattern in blog_patterns:
        if pattern in domain or pattern in path:
            return "blog", f"URL indicates blog '{pattern}'"
    if "blog" in title:
        return "blog", "Title contains 'blog'"
    
    # Rule 6: News patterns
    news_domains = ["news", "bbc.com", "cnn.com", "reuters.com", "nytimes.com", "theguardian.com"]
    for pattern in news_domains:
        if pattern in domain:
            return "news", f"URL indicates news source '{pattern}'"
    
    # Rule 7: Academic signals in content
    academic_signals = [
        ("doi:", "DOI identifier found"),
        ("arxiv:", "arXiv identifier found"),
        ("arxiv.org", "arXiv URL found"),
        ("pmid:", "PubMed ID found"),
        ("pubmed", "PubMed reference found"),
    ]
    for signal, reason in academic_signals:
        if signal in text or signal in url.lower():
            # Conservative: label as 'journal' unless strong paper structure signals
            return "journal", reason
    
    # Rule 8: Strong paper structure signals
    paper_signals = ["abstract", "introduction", "methods", "results", "discussion", "references", "conclusion"]
    signal_count = sum(1 for s in paper_signals if s in text[:5000])  # Check first 5000 chars
    if signal_count >= 3:
        return "paper", f"Academic paper structure detected ({signal_count} signals)"
    
    # Rule 9: Journal/publisher domains (conservative - label as journal, not paper)
    journal_domains = [
        "nature.com", "sciencedirect.com", "springer.com", "wiley.com", 
        "cell.com", "plos.org", "pnas.org", "science.org", "oup.com",
        "tandfonline.com", "mdpi.com", "frontiersin.org", "biomedcentral.com",
        "biorxiv.org", "medrxiv.org", "chemrxiv.org"
    ]
    for jd in journal_domains:
        if jd in domain:
            return "journal", f"Publisher domain '{jd}'"
    
    # Default: other
    return "other", "No matching classification rules"


async def get_or_create_type_info(
    db: AsyncSession,
    entry_id: int,
) -> "EntryType":
    """
    Get or create an EntryType record for the given entry.
    
    Args:
        db: Database session.
        entry_id: The entry ID.
    
    Returns:
        The EntryType instance (either existing or newly created).
    """
    from app.models import EntryType
    
    result = await db.execute(
        select(EntryType).where(EntryType.entry_id == entry_id)
    )
    type_info = result.scalar_one_or_none()
    
    if type_info is None:
        # Create new record
        type_info = EntryType(entry_id=entry_id, effective_type="other")
        db.add(type_info)
        await db.flush()
    
    return type_info


async def update_parse_type(
    db: AsyncSession,
    entry_id: int,
    url: Optional[str],
    title: Optional[str],
    text: Optional[str],
) -> "EntryType":
    """
    Infer and update the parse-based type for an entry.
    
    Args:
        db: Database session.
        entry_id: The entry ID.
        url: Page URL.
        title: Page title.
        text: Page text content.
    
    Returns:
        The updated EntryType instance.
    """
    from app.models import EntryType
    
    # Infer type
    parse_type, parse_reason = infer_parse_type(url, title, text)
    
    # Get or create type info
    type_info = await get_or_create_type_info(db, entry_id)
    
    # Update parse fields
    type_info.parse_type = parse_type
    type_info.parse_reason = parse_reason
    type_info.parse_updated_at = datetime.utcnow()
    
    # Recompute effective type
    type_info.effective_type = resolve_effective_type(
        type_info.user_type,
        type_info.llm_type,
        type_info.parse_type,
    )
    
    await db.flush()
    logger.debug(f"Updated parse type for entry {entry_id}: {parse_type} ({parse_reason})")
    return type_info


async def update_llm_type(
    db: AsyncSession,
    entry_id: int,
    llm_type: str,
    llm_reason: Optional[str] = None,
) -> "EntryType":
    """
    Update the LLM-inferred type for an entry.
    
    Args:
        db: Database session.
        entry_id: The entry ID.
        llm_type: The LLM-inferred type.
        llm_reason: Optional reason (e.g., site_type_reason from LLM).
    
    Returns:
        The updated EntryType instance.
    """
    from app.models import EntryType
    
    # Validate type
    if llm_type not in VALID_ARTICLE_TYPES:
        logger.warning(f"Invalid LLM type '{llm_type}', defaulting to 'other'")
        llm_type = "other"
    
    # Get or create type info
    type_info = await get_or_create_type_info(db, entry_id)
    
    # Update LLM fields
    type_info.llm_type = llm_type
    type_info.llm_reason = llm_reason
    type_info.llm_updated_at = datetime.utcnow()
    
    # Recompute effective type
    type_info.effective_type = resolve_effective_type(
        type_info.user_type,
        type_info.llm_type,
        type_info.parse_type,
    )
    
    await db.flush()
    logger.debug(f"Updated LLM type for entry {entry_id}: {llm_type}")
    return type_info


async def update_user_type(
    db: AsyncSession,
    entry_id: int,
    user_type: Optional[str],
    user_reason: Optional[str] = None,
) -> "EntryType":
    """
    Update the user-overridden type for an entry.
    Pass user_type=None to clear the user override.
    
    Args:
        db: Database session.
        entry_id: The entry ID.
        user_type: The user-set type, or None to clear.
        user_reason: Optional reason from user.
    
    Returns:
        The updated EntryType instance.
    """
    from app.models import EntryType
    
    # Validate type if provided
    if user_type is not None and user_type not in VALID_ARTICLE_TYPES:
        raise ValueError(f"Invalid user type '{user_type}'. Must be one of: {', '.join(sorted(VALID_ARTICLE_TYPES))}")
    
    # Get or create type info
    type_info = await get_or_create_type_info(db, entry_id)
    
    # Update user fields
    type_info.user_type = user_type
    type_info.user_reason = user_reason
    type_info.user_updated_at = datetime.utcnow() if user_type else None
    
    # Recompute effective type
    type_info.effective_type = resolve_effective_type(
        type_info.user_type,
        type_info.llm_type,
        type_info.parse_type,
    )
    
    await db.flush()
    logger.debug(f"Updated user type for entry {entry_id}: {user_type}")
    return type_info
