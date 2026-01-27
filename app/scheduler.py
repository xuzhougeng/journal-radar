"""
APScheduler setup for scheduled journal checks.
"""

import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_runtime_config

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler: AsyncIOScheduler | None = None


async def scheduled_check_job():
    """Job function for scheduled checks."""
    from app.runner import run_check

    logger.info("Starting scheduled check...")
    try:
        result = await run_check()
        logger.info(
            f"Scheduled check completed: {result['new_entries']} new entries, "
            f"{result['notifications']} notifications sent"
        )
    except Exception as e:
        logger.error(f"Scheduled check failed: {e}")


def start_scheduler():
    """Start the APScheduler with configured schedule."""
    global scheduler

    config = get_runtime_config()
    scheduler = AsyncIOScheduler(timezone=config.timezone)

    # Add the daily check job
    scheduler.add_job(
        scheduled_check_job,
        trigger=CronTrigger(
            hour=config.check_hour,
            minute=config.check_minute,
            timezone=config.timezone,
        ),
        id="daily_check",
        name="Daily Journal Check",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        f"Scheduler started. Daily check at {config.check_hour:02d}:{config.check_minute:02d} "
        f"({config.timezone})"
    )


def shutdown_scheduler():
    """Shutdown the scheduler gracefully."""
    global scheduler
    if scheduler:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shutdown")


def reschedule_check_job():
    """
    Reschedule the check job with current config.
    Called when schedule settings are updated.
    """
    global scheduler
    if not scheduler:
        logger.warning("Cannot reschedule: scheduler not running")
        return

    config = get_runtime_config()

    # Update the job's trigger
    scheduler.reschedule_job(
        "daily_check",
        trigger=CronTrigger(
            hour=config.check_hour,
            minute=config.check_minute,
            timezone=config.timezone,
        ),
    )

    logger.info(
        f"Schedule updated. Daily check at {config.check_hour:02d}:{config.check_minute:02d} "
        f"({config.timezone})"
    )


def get_next_run_time() -> str | None:
    """Get the next scheduled run time as ISO format string."""
    global scheduler
    if scheduler:
        job = scheduler.get_job("daily_check")
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
    return None
