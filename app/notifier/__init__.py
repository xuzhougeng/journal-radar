"""Notification modules for pushing alerts to users."""

from app.notifier.bark import BarkNotifier

__all__ = ["BarkNotifier"]
