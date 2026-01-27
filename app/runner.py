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
from app.models import Subscription, Entry, CheckRun, EntryContent
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


async def fetch_exa_content_for_entries(
    session, subscription_id: int, new_entries: list[SourceEntry]
) -> int:
    """
    Fetch web page content via Exa AI for new RSS entries and save to DB.

    Args:
        session: Database session.
        subscription_id: ID of the subscription.
        new_entries: List of newly inserted entries (SourceEntry objects).

    Returns:
        Number of entries with successfully fetched content.
    """
    from app.exa_ai import fetch_contents, truncate_text, is_exa_enabled

    if not is_exa_enabled():
        return 0

    if not new_entries:
        return 0

    # Collect URLs and build fingerprint -> link mapping
    urls = [entry.link for entry in new_entries if entry.link]
    if not urls:
        return 0

    # Fetch content from Exa
    exa_response = await fetch_contents(urls)
    if not exa_response:
        logger.warning(f"Exa content fetch returned no results for subscription {subscription_id}")
        return 0

    # Log any errors
    if exa_response.errors:
        for error in exa_response.errors:
            logger.warning(f"Exa failed for {error.id}: {error.error_tag or 'unknown'}")

    # Build URL -> Exa result mapping (use both original id and returned url)
    url_to_result = {}
    for result in exa_response.results:
        # Map by original requested URL (id)
        if result.id:
            url_to_result[result.id] = result
        # Also map by returned URL if different
        if result.url and result.url != result.id:
            url_to_result[result.url] = result

    # Get entry IDs from database by fingerprint
    fingerprint_to_entry = {entry.fingerprint: entry for entry in new_entries}
    fingerprints = list(fingerprint_to_entry.keys())

    db_result = await session.execute(
        select(Entry.id, Entry.fingerprint, Entry.link)
        .where(Entry.subscription_id == subscription_id)
        .where(Entry.fingerprint.in_(fingerprints))
    )
    db_entries = db_result.all()

    saved_count = 0
    for entry_id, fingerprint, link in db_entries:
        # Find matching Exa result
        exa_result = url_to_result.get(link)
        if not exa_result:
            logger.debug(f"No Exa result found for entry {entry_id} with link {link}")
            continue

        # Truncate text
        truncated_text = truncate_text(exa_result.text)

        # Insert EntryContent (ignore if already exists due to unique constraint)
        stmt = (
            sqlite_insert(EntryContent)
            .values(
                entry_id=entry_id,
                provider="exa",
                request_id=exa_response.request_id,
                status=exa_result.status,
                url=exa_result.url,
                title=exa_result.title,
                author=exa_result.author,
                text=truncated_text,
                raw_path=exa_response.raw_path,
                cost_total=exa_response.cost_total,
                cost_text=exa_response.cost_text,
                search_time_ms=exa_response.search_time_ms,
            )
            .on_conflict_do_nothing(index_elements=["entry_id"])
        )
        result = await session.execute(stmt)
        if result.rowcount > 0:
            saved_count += 1

    await session.commit()
    logger.info(f"Saved Exa content for {saved_count}/{len(new_entries)} entries")
    return saved_count


async def run_check() -> dict[str, Any]:
    """
    Run a check across all enabled subscriptions.

    Returns:
        Dictionary with check results including new_entries count and notifications count.
    """
    from app.notifier.bark import BarkNotifier
    from app.exa_ai import is_exa_enabled

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

                        # Fetch Exa content for RSS subscriptions (if enabled)
                        if subscription.source_type == "rss" and is_exa_enabled():
                            try:
                                exa_count = await fetch_exa_content_for_entries(
                                    session, subscription.id, new_entries
                                )
                                if exa_count > 0:
                                    logger.info(
                                        f"Exa content fetched for {exa_count} entries from '{subscription.name}'"
                                    )
                            except Exception as exa_err:
                                # Log but don't fail the whole run
                                logger.error(
                                    f"Exa content fetch failed for '{subscription.name}': {exa_err}"
                                )

                except Exception as e:
                    error_msg = f"Error processing '{subscription.name}': {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Send notifications for new entries
            for sub_id, new_entries in subscription_new_entries.items():
                sub_result = await session.execute(
                    select(Subscription).where(Subscription.id == sub_id)
                )
                subscription = sub_result.scalar_one()

                try:
                    sent = await notifier.notify_new_entries(
                        subscription.name, new_entries, check_run.id
                    )
                    total_notifications += sent

                    # Mark entries as notified
                    for entry in new_entries:
                        await session.execute(
                            Entry.__table__.update()
                            .where(Entry.fingerprint == entry.fingerprint)
                            .where(Entry.subscription_id == sub_id)
                            .values(notified=True)
                        )
                    await session.commit()

                except Exception as e:
                    logger.error(
                        f"Error sending notification for '{subscription.name}': {e}"
                    )
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
