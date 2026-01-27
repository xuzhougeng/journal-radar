"""
Centralized logging configuration.

Goal:
- Unified log format across the app and Uvicorn.
- Default to INFO level (override with env var LOG_LEVEL).
"""

from __future__ import annotations

import logging
import logging.config
import os

_CONFIGURED = False


def _normalize_level(level: str) -> str:
    lvl = (level or "").strip().upper()
    if not lvl:
        return "INFO"
    valid = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
    return lvl if lvl in valid else "INFO"


def setup_logging(level: str | None = None) -> None:
    """
    Configure Python logging using dictConfig.

    - Default level: INFO
    - Override level: LOG_LEVEL env var (e.g. DEBUG)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = _normalize_level(level or os.getenv("LOG_LEVEL", "INFO"))

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            # Uvicorn access logs are emitted as msg+args; AccessFormatter populates:
            # client_addr, request_line, status_code
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(client_addr)s - \"%(request_line)s\" %(status_code)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "access",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["default"],
        },
        "loggers": {
            # Ensure Uvicorn uses our handlers/formatters and doesn't double-log via root.
            "uvicorn": {"level": log_level, "handlers": ["default"], "propagate": False},
            "uvicorn.error": {
                "level": log_level,
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.asgi": {
                "level": log_level,
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": log_level,
                "handlers": ["access"],
                "propagate": False,
            },
            # Prevent SQLAlchemy from attaching its own default handler (echo mode),
            # and keep formatting consistent when SQL logging is enabled.
            "sqlalchemy.engine": {
                "level": log_level,
                "handlers": ["default"],
                "propagate": False,
            },
            # SQLAlchemy emits engine logs on this logger name when echo=True.
            "sqlalchemy.engine.Engine": {
                "level": log_level,
                "handlers": ["default"],
                "propagate": False,
            },
            "sqlalchemy.pool": {
                "level": log_level,
                "handlers": ["default"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(config)
    _CONFIGURED = True

