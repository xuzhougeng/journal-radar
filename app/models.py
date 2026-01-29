"""
SQLAlchemy models for Journal Monitor.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Subscription(Base):
    """A subscription to a journal source (RSS feed, Crossref ISSN, etc.)."""

    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'rss', 'crossref', etc.
    config: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # JSON string with source-specific config
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    entries: Mapped[list["Entry"]] = relationship(
        "Entry", back_populates="subscription", cascade="all, delete-orphan"
    )


class Entry(Base):
    """A journal entry/article fetched from a source."""

    __tablename__ = "entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subscription_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False
    )
    fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # SHA256 hash for deduplication
    title: Mapped[str] = mapped_column(Text, nullable=False)
    link: Mapped[str] = mapped_column(Text, nullable=False)
    doi: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    authors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    journal_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    notified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    subscription: Mapped["Subscription"] = relationship(
        "Subscription", back_populates="entries"
    )
    content: Mapped[Optional["EntryContent"]] = relationship(
        "EntryContent", back_populates="entry", uselist=False, cascade="all, delete-orphan"
    )
    structured: Mapped[Optional["EntryStructure"]] = relationship(
        "EntryStructure", back_populates="entry", uselist=False, cascade="all, delete-orphan"
    )
    type_info: Mapped[Optional["EntryType"]] = relationship(
        "EntryType", back_populates="entry", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Unique constraint on fingerprint per subscription for deduplication
        Index("ix_entries_fingerprint", "subscription_id", "fingerprint", unique=True),
        # Index for querying recent entries
        Index("ix_entries_fetched_at", "fetched_at"),
    )


class CheckRun(Base):
    """A record of a scheduled or manual check run."""

    __tablename__ = "check_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="running"
    )  # 'running', 'completed', 'failed'
    total_subscriptions: Mapped[int] = mapped_column(Integer, default=0)
    total_new_entries: Mapped[int] = mapped_column(Integer, default=0)
    total_notifications: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Index for querying recent runs
    __table_args__ = (Index("ix_check_runs_started_at", "started_at"),)


class EntryContent(Base):
    """Extracted web page content for an entry (via Exa AI or other providers)."""

    __tablename__ = "entry_contents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entries.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, default="exa"
    )  # 'exa' or future providers
    request_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # Exa requestId for tracking
    status: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # 'success', 'failed', etc.
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # URL returned by Exa
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    author: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Truncated text content
    raw_path: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )  # Relative path to raw JSON (e.g., data/exa/xxx.json)
    cost_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cost_text: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    search_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    entry: Mapped["Entry"] = relationship("Entry", back_populates="content")

    __table_args__ = (
        # Index for querying by provider
        Index("ix_entry_contents_provider", "provider"),
    )


class EntryStructure(Base):
    """LLM-extracted structured information for an entry (site type and summary)."""

    __tablename__ = "entry_structures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entries.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, default="openai_compatible"
    )  # 'openai_compatible' or future providers
    model: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )  # Model name used (e.g., 'gpt-4o-mini')
    base_url: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )  # API base URL for debugging
    site_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="other"
    )  # Enum: paper, journal, news, blog, docs, repository, forum, product, dataset, other
    site_type_reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Reason when site_type is 'other'
    summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # LLM-generated summary
    raw_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Raw LLM response JSON for debugging
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
    )  # 'success', 'failed', 'pending'
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Error message if failed
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    entry: Mapped["Entry"] = relationship("Entry", back_populates="structured")

    __table_args__ = (
        # Index for querying by status
        Index("ix_entry_structures_status", "status"),
    )


class EntryType(Base):
    """
    Unified article type classification for an entry.
    Stores parse-inferred, LLM-inferred, and user-overridden types,
    along with a computed effective_type based on priority: user > llm > parse.
    """

    __tablename__ = "entry_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entries.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Effective (final) type - computed from user > llm > parse priority
    effective_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="other"
    )  # Enum: paper, journal, news, blog, docs, repository, forum, product, dataset, other

    # Parse-inferred type (heuristic rules based on URL/content)
    parse_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    parse_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parse_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # LLM-inferred type (from EntryStructure.site_type)
    llm_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    llm_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # User-overridden type (manual classification)
    user_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    entry: Mapped["Entry"] = relationship("Entry", back_populates="type_info")

    __table_args__ = (
        # Index for querying/filtering by effective type
        Index("ix_entry_types_effective_type", "effective_type"),
    )


class Notification(Base):
    """A record of a push notification sent."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    check_run_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("check_runs.id", ondelete="SET NULL"), nullable=True
    )
    subscription_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("subscriptions.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sent_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class AppConfig(Base):
    """
    Application configuration stored in database.
    Single-row table (id=1) storing runtime and auth config as JSON.
    """

    __tablename__ = "app_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    # Runtime configuration (bark, exa, schedule, push, http settings)
    runtime_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    # Auth configuration (password hash, salt, session secret)
    auth_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
