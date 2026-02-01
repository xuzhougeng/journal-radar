"""
Bark push notification implementation.
https://github.com/Finb/Bark
"""

import logging
from typing import Optional

import httpx

from app.config import get_runtime_config
from app.sources.base import Entry

logger = logging.getLogger(__name__)


class BarkNotifier:
    """Bark push notification sender."""

    def __init__(self):
        config = get_runtime_config()
        self.device_key = config.bark_device_key
        self.server_url = config.bark_server_url.rstrip("/")
        self.timeout = config.request_timeout

    async def send(
        self,
        title: str,
        body: str,
        url: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        """
        Send a push notification via Bark.

        Args:
            title: Notification title
            body: Notification body
            url: Optional URL to open when notification is tapped
            group: Optional group name for grouping notifications

        Returns:
            True if sent successfully, False otherwise
        """
        config = get_runtime_config()
        if not config.bark_enabled:
            logger.debug("Bark push notifications are disabled, skipping")
            return False

        if not self.device_key:
            logger.warning("Bark device key not configured, skipping notification")
            return False

        endpoint = f"{self.server_url}/{self.device_key}"

        payload = {
            "title": title,
            "body": body,
            "group": group or "Journal Monitor",
        }

        if url:
            payload["url"] = url

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()

                result = response.json()
                if result.get("code") == 200:
                    logger.info(f"Notification sent: {title}")
                    return True
                else:
                    logger.error(f"Bark API error: {result}")
                    return False

        except httpx.HTTPStatusError as e:
            logger.error(f"Bark HTTP error: {e.response.status_code} - {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Failed to send Bark notification: {e}")
            return False

    async def notify_digest(
        self,
        journal_counts: dict[str, int],
        check_run_id: Optional[int] = None,
    ) -> int:
        """
        Send a single digest notification summarizing all journal updates.

        Args:
            journal_counts: Dict of journal_name -> new_entry_count (only journals with updates)
            check_run_id: Optional check run ID for logging

        Returns:
            1 if notification sent successfully, 0 otherwise
        """
        if not journal_counts:
            return 0

        config = get_runtime_config()
        if not config.bark_enabled:
            logger.debug("Bark push notifications are disabled, skipping digest notification")
            return 0

        if not self.device_key:
            logger.warning("Bark device key not configured, skipping digest notification")
            return 0

        # Calculate totals
        total_articles = sum(journal_counts.values())
        journal_count = len(journal_counts)

        # Build title
        title = f"期刊更新：{total_articles} 篇（{journal_count} 个期刊）"

        # Build body: sort by count descending, then alphabetically
        config = get_runtime_config()
        max_lines = config.push_max_entries_per_message

        sorted_journals = sorted(
            journal_counts.items(),
            key=lambda x: (-x[1], x[0])  # descending count, then alphabetical
        )

        lines = []
        for i, (journal_name, count) in enumerate(sorted_journals[:max_lines]):
            lines.append(f"{i + 1}) {journal_name}: {count}")

        if len(sorted_journals) > max_lines:
            remaining = len(sorted_journals) - max_lines
            lines.append(f"... and {remaining} more")

        body = "\n".join(lines)

        # Send the digest notification
        sent = await self.send(
            title=title,
            body=body,
            url=None,
            group="Journal Digest",
        )

        if sent:
            await self._record_notification(
                check_run_id=check_run_id,
                title=title,
                body=body,
                url=None,
                success=True,
            )
            return 1

        return 0

    async def notify_new_entries(
        self,
        journal_name: str,
        entries: list[Entry],
        check_run_id: Optional[int] = None,
    ) -> int:
        """
        Send notifications for new entries (legacy per-journal method).

        Note: This method is kept for backward compatibility but is no longer
        used by the main check runner. Use notify_digest() instead.

        Args:
            journal_name: Name of the journal/subscription
            entries: List of new entries to notify about
            check_run_id: Optional check run ID for logging

        Returns:
            Number of notifications sent
        """
        if not entries:
            return 0

        config = get_runtime_config()
        if not config.bark_enabled:
            logger.debug("Bark push notifications are disabled, skipping notifications")
            return 0

        if not self.device_key:
            logger.warning("Bark device key not configured, skipping notifications")
            return 0

        config = get_runtime_config()
        notifications_sent = 0

        if config.push_merge_entries:
            # Merge entries into a single notification
            sent = await self._send_merged_notification(journal_name, entries)
            if sent:
                notifications_sent = 1
                await self._record_notification(
                    check_run_id=check_run_id,
                    title=f"{journal_name}: {len(entries)} new articles",
                    body=self._format_merged_body(entries),
                    url=entries[0].link if entries else None,
                    success=True,
                )
        else:
            # Send individual notifications
            for entry in entries:
                sent = await self._send_single_notification(journal_name, entry)
                if sent:
                    notifications_sent += 1
                    await self._record_notification(
                        check_run_id=check_run_id,
                        title=entry.title,
                        body=f"From {journal_name}",
                        url=entry.link,
                        success=True,
                    )

        return notifications_sent

    async def _send_merged_notification(
        self, journal_name: str, entries: list[Entry]
    ) -> bool:
        """Send a merged notification for multiple entries."""
        title = f"{journal_name}: {len(entries)} new articles"
        body = self._format_merged_body(entries)

        # Use the first entry's link as the notification URL
        url = entries[0].link if entries else None

        return await self.send(
            title=title,
            body=body,
            url=url,
            group=journal_name,
        )

    async def _send_single_notification(
        self, journal_name: str, entry: Entry
    ) -> bool:
        """Send a notification for a single entry."""
        title = entry.title
        body = f"From {journal_name}"
        if entry.authors:
            body += f"\n{entry.authors}"

        return await self.send(
            title=title,
            body=body,
            url=entry.link,
            group=journal_name,
        )

    def _format_merged_body(self, entries: list[Entry]) -> str:
        """Format multiple entries into a notification body."""
        config = get_runtime_config()
        lines = []
        max_entries = config.push_max_entries_per_message

        for i, entry in enumerate(entries[:max_entries]):
            # Truncate title if too long
            title = entry.title
            if len(title) > 80:
                title = title[:77] + "..."
            lines.append(f"{i + 1}. {title}")

        if len(entries) > max_entries:
            lines.append(f"... and {len(entries) - max_entries} more")

        return "\n".join(lines)

    async def _record_notification(
        self,
        check_run_id: Optional[int],
        title: str,
        body: str,
        url: Optional[str],
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a notification in the database."""
        from app.db import async_session
        from app.models import Notification

        try:
            async with async_session() as session:
                notification = Notification(
                    check_run_id=check_run_id,
                    title=title,
                    body=body,
                    url=url,
                    success=success,
                    error_message=error_message,
                )
                session.add(notification)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to record notification: {e}")
