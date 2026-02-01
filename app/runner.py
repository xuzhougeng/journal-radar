"""
Check runner - coordinates fetching from sources and sending notifications.
"""

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.config import get_runtime_config
from app.db import async_session
from app.models import Subscription, Entry, CheckRun, EntryContent, EntryStructure
from app.sources.base import Entry as SourceEntry

logger = logging.getLogger(__name__)


async def get_source_for_subscription(subscription: Subscription):
    """Get the appropriate source handler for a subscription."""
    config = json.loads(subscription.config)

    if subscription.source_type == "rss":
        from app.sources.rss import RSSSource

        return RSSSource(subscription.id, config)
    elif subscription.source_type == "crossref":
        from app.sources.crossref import CrossrefSource

        return CrossrefSource(subscription.id, config)
    else:
        raise ValueError(f"Unknown source type: {subscription.source_type}")


async def save_new_entries(
    session, subscription_id: int, entries: list[SourceEntry]
) -> list[SourceEntry]:
    """
    Save entries to database, returning only the new ones (not duplicates).
    Uses INSERT OR IGNORE for deduplication based on fingerprint.
    """
    new_entries = []

    for entry in entries:
        # Try to insert, ignoring if fingerprint already exists
        stmt = (
            sqlite_insert(Entry)
            .values(
                subscription_id=subscription_id,
                fingerprint=entry.fingerprint,
                title=entry.title,
                link=entry.link,
                doi=entry.doi,
                authors=entry.authors,
                abstract=entry.abstract,
                journal_name=entry.journal_name,
                published_at=entry.published_at,
            )
            .on_conflict_do_nothing(index_elements=["subscription_id", "fingerprint"])
        )
        result = await session.execute(stmt)

        # If rowcount > 0, it was a new entry
        if result.rowcount > 0:
            new_entries.append(entry)

    await session.commit()
    return new_entries


async def fetch_parse_content_for_entries(
    session, subscription_id: int, new_entries: list[SourceEntry]
) -> list[int]:
    """
    Fetch web page content for new RSS entries using parse providers with fallback.

    Args:
        session: Database session.
        subscription_id: ID of the subscription.
        new_entries: List of newly inserted entries (SourceEntry objects).

    Returns:
        List of entry IDs with successfully fetched content.
    """
    from app.parse import fetch_content_with_fallback, is_parse_enabled
    from app.parse.runner import truncate_text

    if not is_parse_enabled():
        return []

    if not new_entries:
        return []

    # Collect URLs and build fingerprint -> link mapping
    urls = [entry.link for entry in new_entries if entry.link]
    if not urls:
        return []

    config = get_runtime_config()

    # Fetch content with fallback
    parse_results = await fetch_content_with_fallback(urls)

    if not parse_results:
        logger.warning(f"Parse content fetch returned no results for subscription {subscription_id}")
        return []

    # Get entry IDs from database by fingerprint
    fingerprint_to_entry = {entry.fingerprint: entry for entry in new_entries}
    fingerprints = list(fingerprint_to_entry.keys())

    db_result = await session.execute(
        select(Entry.id, Entry.fingerprint, Entry.link)
        .where(Entry.subscription_id == subscription_id)
        .where(Entry.fingerprint.in_(fingerprints))
    )
    db_entries = db_result.all()

    saved_entry_ids = []
    for entry_id, fingerprint, link in db_entries:
        # Find matching parse result
        parse_result = parse_results.get(link)
        if not parse_result:
            logger.debug(f"No parse result found for entry {entry_id} with link {link}")
            continue

        # Check if successful and meets threshold
        if not parse_result.success or not parse_result.meets_threshold(config.parse_min_text_chars):
            logger.debug(
                f"Parse result for entry {entry_id} failed or below threshold: "
                f"success={parse_result.success}, text_len={len(parse_result.text or '')}"
            )
            continue

        # Truncate text
        truncated_text = truncate_text(parse_result.text)

        # Extract metadata for Exa-compatible fields
        metadata = parse_result.metadata or {}

        # Insert EntryContent (ignore if already exists due to unique constraint)
        stmt = (
            sqlite_insert(EntryContent)
            .values(
                entry_id=entry_id,
                provider=parse_result.provider,
                request_id=metadata.get("request_id"),
                status=parse_result.status,
                url=parse_result.final_url,
                title=parse_result.title,
                author=parse_result.author,
                text=truncated_text,
                raw_path=parse_result.raw_path,
                cost_total=metadata.get("cost_total"),
                cost_text=metadata.get("cost_text"),
                search_time_ms=metadata.get("search_time_ms"),
            )
            .on_conflict_do_nothing(index_elements=["entry_id"])
        )
        result = await session.execute(stmt)
        if result.rowcount > 0:
            saved_entry_ids.append(entry_id)
            logger.debug(
                f"Saved content for entry {entry_id} via {parse_result.provider} "
                f"(text: {len(truncated_text or '')} chars)"
            )
            
            # Infer and update parse type
            from app.entry_type import update_parse_type
            await update_parse_type(
                session,
                entry_id,
                url=parse_result.final_url or link,
                title=parse_result.title,
                text=truncated_text,
            )

    await session.commit()
    logger.info(f"Saved parse content for {len(saved_entry_ids)}/{len(new_entries)} entries")
    return saved_entry_ids


# Keep old function name as alias for backward compatibility
async def fetch_exa_content_for_entries(
    session, subscription_id: int, new_entries: list[SourceEntry]
) -> list[int]:
    """
    Legacy alias for fetch_parse_content_for_entries.
    Deprecated: Use fetch_parse_content_for_entries instead.
    """
    return await fetch_parse_content_for_entries(session, subscription_id, new_entries)


async def fetch_llm_structure_for_entries(
    session, entry_ids: list[int]
) -> int:
    """
    Extract structured information via LLM for entries with content.

    Args:
        session: Database session.
        entry_ids: List of entry IDs to process.

    Returns:
        Number of entries with successfully extracted structure.
    """
    from app.llm_struct import is_llm_enabled, extract_structured_info

    if not is_llm_enabled():
        return 0

    if not entry_ids:
        return 0

    config = get_runtime_config()
    saved_count = 0

    for entry_id in entry_ids:
        try:
            # Check if structure already exists (skip if so)
            existing = await session.execute(
                select(EntryStructure).where(EntryStructure.entry_id == entry_id)
            )
            if existing.scalar_one_or_none():
                logger.debug(f"Structure already exists for entry {entry_id}, skipping")
                continue

            # Get the content for this entry
            content_result = await session.execute(
                select(EntryContent).where(EntryContent.entry_id == entry_id)
            )
            content = content_result.scalar_one_or_none()
            if not content or not content.text:
                logger.debug(f"No content for entry {entry_id}, skipping LLM")
                continue

            # Extract structured info
            result = await extract_structured_info(
                title=content.title,
                url=content.url,
                text=content.text,
            )

            # Save to database
            stmt = (
                sqlite_insert(EntryStructure)
                .values(
                    entry_id=entry_id,
                    provider="openai_compatible",
                    model=result.model,
                    base_url=config.llm_base_url,
                    site_type=result.site_type,
                    site_type_reason=result.site_type_reason,
                    summary=result.summary,
                    raw_json=result.raw_json,
                    status="success",
                )
                .on_conflict_do_nothing(index_elements=["entry_id"])
            )
            insert_result = await session.execute(stmt)
            if insert_result.rowcount > 0:
                saved_count += 1
                logger.debug(f"Saved LLM structure for entry {entry_id}: {result.site_type}")
                
                # Sync LLM type to EntryType
                from app.entry_type import update_llm_type
                await update_llm_type(
                    session,
                    entry_id,
                    llm_type=result.site_type,
                    llm_reason=result.site_type_reason,
                )

        except Exception as e:
            logger.warning(f"LLM extraction failed for entry {entry_id}: {e}")
            # Save failed status
            try:
                stmt = (
                    sqlite_insert(EntryStructure)
                    .values(
                        entry_id=entry_id,
                        provider="openai_compatible",
                        model=config.llm_model,
                        base_url=config.llm_base_url,
                        site_type="other",
                        site_type_reason="LLM extraction failed",
                        status="failed",
                        error_message=str(e)[:1000],
                    )
                    .on_conflict_do_nothing(index_elements=["entry_id"])
                )
                await session.execute(stmt)
            except Exception:
                pass

    await session.commit()
    if saved_count > 0:
        logger.info(f"Saved LLM structure for {saved_count}/{len(entry_ids)} entries")
    return saved_count


async def run_check() -> dict[str, Any]:
    """
    Run a check across all enabled subscriptions.

    Returns:
        Dictionary with check results including new_entries count and notifications count.
    """
    from app.notifier.bark import BarkNotifier
    from app.parse import is_parse_enabled

    async with async_session() as session:
        # Create a check run record
        check_run = CheckRun(status="running")
        session.add(check_run)
        await session.commit()
        await session.refresh(check_run)

        total_new_entries = 0
        total_notifications = 0
        errors = []

        try:
            # Get all enabled subscriptions
            result = await session.execute(
                select(Subscription).where(Subscription.enabled == True)
            )
            subscriptions = result.scalars().all()
            check_run.total_subscriptions = len(subscriptions)

            notifier = BarkNotifier()
            subscription_new_entries: dict[int, list[SourceEntry]] = {}

            # Process each subscription
            for subscription in subscriptions:
                try:
                    source = await get_source_for_subscription(subscription)
                    entries = await source.fetch_entries()

                    # Save and get only new entries
                    new_entries = await save_new_entries(
                        session, subscription.id, entries
                    )

                    if new_entries:
                        subscription_new_entries[subscription.id] = new_entries
                        total_new_entries += len(new_entries)
                        logger.info(
                            f"Found {len(new_entries)} new entries for '{subscription.name}'"
                        )

                        # Fetch content for RSS subscriptions using parse providers (if enabled)
                        if subscription.source_type == "rss" and is_parse_enabled():
                            try:
                                content_entry_ids = await fetch_parse_content_for_entries(
                                    session, subscription.id, new_entries
                                )
                                if content_entry_ids:
                                    logger.info(
                                        f"Content fetched for {len(content_entry_ids)} entries from '{subscription.name}'"
                                    )
                                    
                                    # Auto-trigger LLM structured extraction (if enabled)
                                    from app.llm_struct import is_llm_enabled
                                    llm_config = get_runtime_config()
                                    if is_llm_enabled() and llm_config.llm_auto_extract:
                                        try:
                                            llm_count = await fetch_llm_structure_for_entries(
                                                session, content_entry_ids
                                            )
                                            if llm_count > 0:
                                                logger.info(
                                                    f"LLM structure extracted for {llm_count} entries from '{subscription.name}'"
                                                )
                                        except Exception as llm_err:
                                            # Log but don't fail the whole run
                                            logger.error(
                                                f"LLM extraction failed for '{subscription.name}': {llm_err}"
                                            )
                                    elif is_llm_enabled() and not llm_config.llm_auto_extract:
                                        logger.debug(
                                            f"LLM auto-extraction disabled, skipping for '{subscription.name}'"
                                        )
                            except Exception as parse_err:
                                # Log but don't fail the whole run
                                logger.error(
                                    f"Content fetch failed for '{subscription.name}': {parse_err}"
                                )

                except Exception as e:
                    error_msg = f"Error processing '{subscription.name}': {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Send a single digest notification for all new entries
            config = get_runtime_config()
            if subscription_new_entries:
                if not config.bark_enabled:
                    logger.info("Bark push notifications are disabled, skipping digest notification")
                elif not config.bark_device_key:
                    logger.info("Bark device key not configured, skipping digest notification")
                else:
                    try:
                        # Build journal_counts: name -> count
                        journal_counts: dict[str, int] = {}

                        for sub_id, new_entries in subscription_new_entries.items():
                            sub_result = await session.execute(
                                select(Subscription).where(Subscription.id == sub_id)
                            )
                            subscription = sub_result.scalar_one()
                            journal_counts[subscription.name] = len(new_entries)

                        # Send one digest notification
                        sent = await notifier.notify_digest(journal_counts, check_run.id)
                        total_notifications = sent

                        # Mark all entries as notified
                        for sub_id, new_entries in subscription_new_entries.items():
                            for entry in new_entries:
                                await session.execute(
                                    Entry.__table__.update()
                                    .where(Entry.fingerprint == entry.fingerprint)
                                    .where(Entry.subscription_id == sub_id)
                                    .values(notified=True)
                                )
                        await session.commit()

                    except Exception as e:
                        logger.error(f"Error sending digest notification: {e}")
                        errors.append(str(e))

            # Update check run record
            check_run.completed_at = datetime.utcnow()
            check_run.status = "completed" if not errors else "completed_with_errors"
            check_run.total_new_entries = total_new_entries
            check_run.total_notifications = total_notifications
            if errors:
                check_run.error_message = "\n".join(errors)

            await session.commit()

        except Exception as e:
            check_run.completed_at = datetime.utcnow()
            check_run.status = "failed"
            check_run.error_message = str(e)
            await session.commit()
            logger.error(f"Check run failed: {e}")
            raise

        return {
            "run_id": check_run.id,
            "status": check_run.status,
            "subscriptions": check_run.total_subscriptions,
            "new_entries": total_new_entries,
            "notifications": total_notifications,
            "errors": errors,
        }
