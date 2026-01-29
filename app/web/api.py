"""
API routes for Journal Monitor.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import Response

from app.db import get_db
from app.models import Subscription, Entry, CheckRun, Notification
from app.config import get_runtime_config, StaticConfig
from app.web.auth import require_login_api

router = APIRouter()


# Pydantic models for request/response
class SubscriptionCreate(BaseModel):
    name: str
    source_type: str  # 'rss' or 'crossref'
    config: dict  # Source-specific config


class SubscriptionResponse(BaseModel):
    id: int
    name: str
    source_type: str
    config: dict
    enabled: bool

    class Config:
        from_attributes = True


class EntryResponse(BaseModel):
    id: int
    title: str
    link: str
    doi: Optional[str]
    authors: Optional[str]
    journal_name: Optional[str]
    published_at: Optional[str]
    fetched_at: str
    subscription_id: int

    class Config:
        from_attributes = True


class CheckRunResponse(BaseModel):
    id: int
    started_at: str
    completed_at: Optional[str]
    status: str
    total_subscriptions: int
    total_new_entries: int
    total_notifications: int
    error_message: Optional[str]

    class Config:
        from_attributes = True


class PushTestRequest(BaseModel):
    title: str = "Test Notification"
    body: str = "This is a test from Journal Monitor"


# Settings API models
class SettingsResponse(BaseModel):
    """Response model for settings (excludes sensitive data)."""
    # Bark
    bark_configured: bool
    bark_server_url: str
    # Exa
    exa_configured: bool
    exa_livecrawl: str
    exa_livecrawl_timeout_ms: int
    exa_text_max_chars: int
    exa_contents_retries: int
    # Parse/Content fetching
    parse_providers_order: list[str]
    parse_min_text_chars: int
    # LLM
    llm_configured: bool
    llm_base_url: str
    llm_model: str
    llm_timeout: int
    llm_max_input_chars: int
    # Schedule
    check_hour: int
    check_minute: int
    timezone: str
    # Push
    push_merge_entries: bool
    push_max_entries_per_message: int
    # HTTP
    request_timeout: int
    user_agent: str
    # Auth
    auth_username: str


class SettingsUpdate(BaseModel):
    """Request model for updating settings."""
    # Bark (optional - only update if provided)
    bark_device_key: Optional[str] = None
    bark_server_url: Optional[str] = None
    # Exa
    exa_api_key: Optional[str] = None
    exa_livecrawl: Optional[str] = Field(
        default=None,
        pattern="^(fallback|always|never)$",
    )
    exa_livecrawl_timeout_ms: Optional[int] = Field(default=None, ge=1000, le=120000)
    exa_text_max_chars: Optional[int] = Field(default=None, ge=1000, le=500000)
    exa_contents_retries: Optional[int] = Field(default=None, ge=0, le=10)
    # Parse/Content fetching
    parse_providers_order: Optional[list[str]] = None
    parse_min_text_chars: Optional[int] = Field(default=None, ge=0, le=10000)
    # LLM
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_timeout: Optional[int] = Field(default=None, ge=10, le=300)
    llm_max_input_chars: Optional[int] = Field(default=None, ge=1000, le=100000)
    # Schedule
    check_hour: Optional[int] = Field(default=None, ge=0, le=23)
    check_minute: Optional[int] = Field(default=None, ge=0, le=59)
    timezone: Optional[str] = None
    # Push
    push_merge_entries: Optional[bool] = None
    push_max_entries_per_message: Optional[int] = Field(default=None, ge=1, le=50)
    # HTTP
    request_timeout: Optional[int] = Field(default=None, ge=5, le=300)
    user_agent: Optional[str] = None
    # Auth
    auth_username: Optional[str] = None


class PasswordUpdate(BaseModel):
    """Request model for setting a custom password."""
    password: str = Field(min_length=8, max_length=128)


def _canonical_json(obj: object) -> str:
    """Stable JSON encoding for comparisons and exports."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# =============================================================================
# Settings API (requires login)
# =============================================================================

@router.get("/settings", response_model=SettingsResponse)
async def get_settings(
    request: Request,
    _: None = Depends(require_login_api),
):
    """Get current application settings (sensitive values masked)."""
    config = get_runtime_config()
    
    return SettingsResponse(
        bark_configured=bool(config.bark_device_key),
        bark_server_url=config.bark_server_url,
        exa_configured=bool(config.exa_api_key),
        exa_livecrawl=config.exa_livecrawl,
        exa_livecrawl_timeout_ms=config.exa_livecrawl_timeout_ms,
        exa_text_max_chars=config.exa_text_max_chars,
        exa_contents_retries=config.exa_contents_retries,
        parse_providers_order=config.parse_providers_order,
        parse_min_text_chars=config.parse_min_text_chars,
        llm_configured=bool(config.llm_api_key),
        llm_base_url=config.llm_base_url,
        llm_model=config.llm_model,
        llm_timeout=config.llm_timeout,
        llm_max_input_chars=config.llm_max_input_chars,
        check_hour=config.check_hour,
        check_minute=config.check_minute,
        timezone=config.timezone,
        push_merge_entries=config.push_merge_entries,
        push_max_entries_per_message=config.push_max_entries_per_message,
        request_timeout=config.request_timeout,
        user_agent=config.user_agent,
        auth_username=config.auth_username,
    )


@router.put("/settings")
async def update_settings(
    request: Request,
    data: SettingsUpdate,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Update application settings."""
    from app.config_store import update_runtime_config
    from app.scheduler import reschedule_check_job
    
    # Build updates dict with only provided fields
    updates = {}
    schedule_changed = False
    
    if data.bark_device_key is not None:
        # Empty string means "clear the key"
        updates["bark_device_key"] = data.bark_device_key if data.bark_device_key else None
    if data.bark_server_url is not None:
        updates["bark_server_url"] = data.bark_server_url
    if data.exa_api_key is not None:
        updates["exa_api_key"] = data.exa_api_key if data.exa_api_key else None
    if data.exa_livecrawl is not None:
        updates["exa_livecrawl"] = data.exa_livecrawl
    if data.exa_livecrawl_timeout_ms is not None:
        updates["exa_livecrawl_timeout_ms"] = data.exa_livecrawl_timeout_ms
    if data.exa_text_max_chars is not None:
        updates["exa_text_max_chars"] = data.exa_text_max_chars
    if data.exa_contents_retries is not None:
        updates["exa_contents_retries"] = data.exa_contents_retries
    if data.parse_providers_order is not None:
        # Validate providers
        valid_providers = {"exa", "direct"}
        for p in data.parse_providers_order:
            if p not in valid_providers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid provider: {p}. Valid providers: {', '.join(valid_providers)}"
                )
        updates["parse_providers_order"] = data.parse_providers_order
    if data.parse_min_text_chars is not None:
        updates["parse_min_text_chars"] = data.parse_min_text_chars
    if data.llm_api_key is not None:
        updates["llm_api_key"] = data.llm_api_key if data.llm_api_key else None
    if data.llm_base_url is not None:
        updates["llm_base_url"] = data.llm_base_url
    if data.llm_model is not None:
        updates["llm_model"] = data.llm_model
    if data.llm_timeout is not None:
        updates["llm_timeout"] = data.llm_timeout
    if data.llm_max_input_chars is not None:
        updates["llm_max_input_chars"] = data.llm_max_input_chars
    if data.check_hour is not None:
        updates["check_hour"] = data.check_hour
        schedule_changed = True
    if data.check_minute is not None:
        updates["check_minute"] = data.check_minute
        schedule_changed = True
    if data.timezone is not None:
        updates["timezone"] = data.timezone
        schedule_changed = True
    if data.push_merge_entries is not None:
        updates["push_merge_entries"] = data.push_merge_entries
    if data.push_max_entries_per_message is not None:
        updates["push_max_entries_per_message"] = data.push_max_entries_per_message
    if data.request_timeout is not None:
        updates["request_timeout"] = data.request_timeout
    if data.user_agent is not None:
        updates["user_agent"] = data.user_agent
    if data.auth_username is not None:
        updates["auth_username"] = data.auth_username
    
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    # Apply updates
    new_config = await update_runtime_config(db, updates)
    
    # Reschedule if schedule changed
    if schedule_changed:
        reschedule_check_job()
    
    return {
        "status": "updated",
        "updated_fields": list(updates.keys()),
    }


@router.post("/settings/password/rotate")
async def rotate_password(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Generate a new random admin password."""
    from app.config_store import rotate_admin_password
    
    new_password = await rotate_admin_password(db)
    
    return {
        "status": "rotated",
        "password": new_password,
        "message": "Save this password! It will not be shown again.",
    }


@router.post("/settings/password")
async def set_password(
    request: Request,
    data: PasswordUpdate,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Set a custom admin password."""
    from app.config_store import update_auth_password
    
    await update_auth_password(db, data.password)
    
    return {
        "status": "updated",
        "message": "Password updated successfully.",
    }


# =============================================================================
# Subscriptions API (all require login)
# =============================================================================

@router.get("/subscriptions", response_model=list[SubscriptionResponse])
async def list_subscriptions(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """List all subscriptions."""
    result = await db.execute(select(Subscription).order_by(desc(Subscription.created_at)))
    subscriptions = result.scalars().all()

    return [
        SubscriptionResponse(
            id=s.id,
            name=s.name,
            source_type=s.source_type,
            config=json.loads(s.config),
            enabled=s.enabled,
        )
        for s in subscriptions
    ]


@router.post("/subscriptions", response_model=SubscriptionResponse)
async def create_subscription(
    request: Request,
    data: SubscriptionCreate,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Create a new subscription."""
    # Validate source type
    if data.source_type not in ["rss", "crossref"]:
        raise HTTPException(status_code=400, detail="Invalid source_type. Must be 'rss' or 'crossref'")

    # Validate config based on source type
    if data.source_type == "rss" and "feed_url" not in data.config:
        raise HTTPException(status_code=400, detail="RSS source requires 'feed_url' in config")
    if data.source_type == "crossref" and "issn" not in data.config:
        raise HTTPException(status_code=400, detail="Crossref source requires 'issn' in config")

    subscription = Subscription(
        name=data.name,
        source_type=data.source_type,
        config=json.dumps(data.config),
    )
    db.add(subscription)
    await db.commit()
    await db.refresh(subscription)

    return SubscriptionResponse(
        id=subscription.id,
        name=subscription.name,
        source_type=subscription.source_type,
        config=data.config,
        enabled=subscription.enabled,
    )


@router.delete("/subscriptions/{subscription_id}")
async def delete_subscription(
    request: Request,
    subscription_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Delete a subscription."""
    result = await db.execute(
        select(Subscription).where(Subscription.id == subscription_id)
    )
    subscription = result.scalar_one_or_none()

    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    await db.delete(subscription)
    await db.commit()

    return {"status": "deleted", "id": subscription_id}


@router.patch("/subscriptions/{subscription_id}/toggle")
async def toggle_subscription(
    request: Request,
    subscription_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Toggle subscription enabled/disabled."""
    result = await db.execute(
        select(Subscription).where(Subscription.id == subscription_id)
    )
    subscription = result.scalar_one_or_none()

    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    subscription.enabled = not subscription.enabled
    await db.commit()

    return {"status": "toggled", "id": subscription_id, "enabled": subscription.enabled}


@router.get("/subscriptions/export")
async def export_subscriptions(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Export all subscriptions as a downloadable JSON file."""
    result = await db.execute(select(Subscription).order_by(desc(Subscription.created_at)))
    subscriptions = result.scalars().all()

    payload = {
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "subscriptions": [
            {
                "name": s.name,
                "source_type": s.source_type,
                "config": json.loads(s.config),
                "enabled": bool(s.enabled),
            }
            for s in subscriptions
        ],
    }

    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    filename = f"journal-monitor-subscriptions-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    return Response(
        content=data,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/subscriptions/import")
async def import_subscriptions(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Import subscriptions from a JSON export file."""
    raw = await file.read()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")

    if isinstance(parsed, dict) and "subscriptions" in parsed:
        items = parsed.get("subscriptions")
    else:
        items = parsed

    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="Invalid format: expected a list of subscriptions")

    # Existing subscriptions (dedupe by source_type + canonical config)
    existing_result = await db.execute(select(Subscription.source_type, Subscription.config))
    existing_keys: set[tuple[str, str]] = set()
    for source_type, config_str in existing_result.all():
        try:
            existing_keys.add((source_type, _canonical_json(json.loads(config_str))))
        except Exception:
            # If existing row has bad JSON, fall back to raw string comparison
            existing_keys.add((source_type, config_str))

    created = 0
    skipped = 0
    errors: list[dict] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append({"index": idx, "error": "Item is not an object"})
            continue

        name = item.get("name")
        source_type = item.get("source_type")
        config = item.get("config")
        enabled = item.get("enabled", True)

        if not isinstance(name, str) or not name.strip():
            errors.append({"index": idx, "error": "Missing/invalid 'name'"})
            continue
        if source_type not in ["rss", "crossref"]:
            errors.append({"index": idx, "error": "Invalid 'source_type' (must be 'rss' or 'crossref')"})
            continue
        if not isinstance(config, dict):
            errors.append({"index": idx, "error": "Missing/invalid 'config' (must be object)"})
            continue
        if source_type == "rss" and "feed_url" not in config:
            errors.append({"index": idx, "error": "RSS requires 'feed_url' in config"})
            continue
        if source_type == "crossref" and "issn" not in config:
            errors.append({"index": idx, "error": "Crossref requires 'issn' in config"})
            continue

        key = (source_type, _canonical_json(config))
        if key in existing_keys:
            skipped += 1
            continue

        subscription = Subscription(
            name=name.strip(),
            source_type=source_type,
            config=json.dumps(config, ensure_ascii=False),
            enabled=bool(enabled),
        )
        db.add(subscription)
        existing_keys.add(key)
        created += 1

    await db.commit()

    return {
        "status": "ok",
        "created": created,
        "skipped": skipped,
        "errors": errors,
    }


# =============================================================================
# Entries API
# =============================================================================

@router.get("/entries")
async def list_entries(
    limit: int = 50,
    offset: int = 0,
    subscription_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """List recent entries."""
    query = select(Entry).order_by(desc(Entry.fetched_at))

    if subscription_id:
        query = query.where(Entry.subscription_id == subscription_id)

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    entries = result.scalars().all()

    return [
        {
            "id": e.id,
            "title": e.title,
            "link": e.link,
            "doi": e.doi,
            "authors": e.authors,
            "journal_name": e.journal_name,
            "published_at": e.published_at.isoformat() if e.published_at else None,
            "fetched_at": e.fetched_at.isoformat(),
            "subscription_id": e.subscription_id,
            "notified": e.notified,
        }
        for e in entries
    ]


# =============================================================================
# Check runs API (requires login)
# =============================================================================

@router.get("/runs")
async def list_runs(
    request: Request,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """List recent check runs."""
    result = await db.execute(
        select(CheckRun).order_by(desc(CheckRun.started_at)).limit(limit)
    )
    runs = result.scalars().all()

    return [
        {
            "id": r.id,
            "started_at": r.started_at.isoformat(),
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "status": r.status,
            "total_subscriptions": r.total_subscriptions,
            "total_new_entries": r.total_new_entries,
            "total_notifications": r.total_notifications,
            "error_message": r.error_message,
        }
        for r in runs
    ]


@router.post("/check/run")
async def trigger_check(
    request: Request,
    _: None = Depends(require_login_api),
):
    """Manually trigger a check run. Requires login."""
    from app.runner import run_check

    try:
        result = await run_check()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Push notification API (requires login)
# =============================================================================

@router.post("/push/test")
async def test_push(
    request: Request,
    data: PushTestRequest,
    _: None = Depends(require_login_api),
):
    """Test Bark push notification. Requires login."""
    config = get_runtime_config()
    
    if not config.bark_device_key:
        raise HTTPException(
            status_code=400,
            detail="Bark device key not configured. Update it in Settings.",
        )

    from app.notifier.bark import BarkNotifier

    notifier = BarkNotifier()
    success = await notifier.send(title=data.title, body=data.body)

    if success:
        return {"status": "sent", "title": data.title}
    else:
        raise HTTPException(status_code=500, detail="Failed to send notification")


@router.post("/parse/test")
async def test_parse(
    request: Request,
    _: None = Depends(require_login_api),
):
    """Test parse providers with a sample URL. Requires login."""
    from app.parse import fetch_content_with_fallback, is_parse_enabled

    config = get_runtime_config()

    if not is_parse_enabled():
        raise HTTPException(
            status_code=400,
            detail="No parse providers enabled. Configure Exa API key or enable direct fetch in Settings.",
        )

    test_url = "https://example.com/"
    results = await fetch_content_with_fallback([test_url])

    if results and test_url in results:
        result = results[test_url]
        return {
            "status": "success" if result.success else "failed",
            "provider": result.provider,
            "url": result.final_url or result.url,
            "title": result.title,
            "text_length": len(result.text) if result.text else 0,
            "raw_path": result.raw_path,
            "error": result.error,
            "metadata": result.metadata,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch content from any provider")


@router.post("/exa/test")
async def test_exa(
    request: Request,
    _: None = Depends(require_login_api),
):
    """Test Exa AI API with a sample URL. Requires login. (Deprecated: use /api/parse/test)"""
    config = get_runtime_config()
    
    if not config.exa_api_key:
        raise HTTPException(
            status_code=400,
            detail="Exa API key not configured. Update it in Settings.",
        )

    from app.exa_ai import fetch_contents

    test_url = "https://example.com/"
    result = await fetch_contents([test_url])

    if result and result.results:
        exa_result = result.results[0]
        return {
            "status": "success",
            "provider": "exa",
            "deprecated": True,
            "message": "Use /api/parse/test instead",
            "request_id": result.request_id,
            "url": exa_result.url,
            "title": exa_result.title,
            "text_length": len(exa_result.text) if exa_result.text else 0,
            "cost_total": result.cost_total,
            "search_time_ms": result.search_time_ms,
            "raw_path": result.raw_path,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch content from Exa API")


@router.post("/llm/test")
async def test_llm(
    request: Request,
    _: None = Depends(require_login_api),
):
    """Test LLM API connection. Requires login."""
    config = get_runtime_config()
    
    if not config.llm_api_key:
        raise HTTPException(
            status_code=400,
            detail="LLM API key not configured. Update it in Settings.",
        )

    from app.llm_struct import test_llm_connection

    try:
        result = await test_llm_connection()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM test failed: {str(e)}")


@router.post("/parse/fetch/{entry_id}")
async def fetch_parse_for_entry(
    request: Request,
    entry_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Manually fetch content for a specific entry using parse providers with fallback. Requires login."""
    import logging
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.parse import fetch_content_with_fallback, is_parse_enabled
    from app.parse.runner import truncate_text
    from app.models import EntryContent

    logger = logging.getLogger(__name__)
    config = get_runtime_config()

    if not is_parse_enabled():
        raise HTTPException(
            status_code=400,
            detail="No parse providers enabled. Configure Exa API key or enable direct fetch in Settings.",
        )

    # Get the entry
    result = await db.execute(select(Entry).where(Entry.id == entry_id))
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    # Check if content already exists
    existing = await db.execute(
        select(EntryContent).where(EntryContent.entry_id == entry_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Content already exists for this entry")

    try:
        # Fetch with fallback
        parse_results = await fetch_content_with_fallback([entry.link])

        if not parse_results or entry.link not in parse_results:
            raise HTTPException(status_code=502, detail=f"No parse result for URL: {entry.link}")

        parse_result = parse_results[entry.link]

        # Check if successful and meets threshold
        if not parse_result.success:
            raise HTTPException(
                status_code=502,
                detail=f"Parse failed ({parse_result.provider}): {parse_result.error or 'unknown error'}"
            )

        if not parse_result.meets_threshold(config.parse_min_text_chars):
            raise HTTPException(
                status_code=502,
                detail=f"Extracted text too short ({len(parse_result.text or '')} chars, min: {config.parse_min_text_chars})"
            )

        truncated_text = truncate_text(parse_result.text)
        metadata = parse_result.metadata or {}

        # Save to database
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
        await db.execute(stmt)
        await db.commit()

        return {
            "status": "success",
            "entry_id": entry_id,
            "provider": parse_result.provider,
            "text_length": len(truncated_text) if truncated_text else 0,
            "raw_path": parse_result.raw_path,
            "metadata": metadata,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parse fetch failed for entry {entry_id} ({entry.link}): {e}")
        raise HTTPException(status_code=500, detail=f"Parse fetch failed: {str(e)}")


@router.post("/exa/fetch/{entry_id}")
async def fetch_exa_for_entry(
    request: Request,
    entry_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Manually fetch content for a specific entry. (Deprecated: use /api/parse/fetch/{entry_id})"""
    # Redirect to the new parse endpoint
    result = await fetch_parse_for_entry(request, entry_id, db, _)
    result["deprecated"] = True
    result["message"] = "Use /api/parse/fetch/{entry_id} instead"
    return result


@router.get("/entries/{entry_id}/parse/preview")
async def get_parse_preview(
    request: Request,
    entry_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Get parsed content for preview sidebar. Supports multiple providers. Requires login."""
    from pathlib import Path
    from sqlalchemy.orm import selectinload
    from app.models import EntryContent

    # Get the entry with its content
    result = await db.execute(
        select(Entry).where(Entry.id == entry_id).options(selectinload(Entry.content))
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    if not entry.content:
        raise HTTPException(status_code=404, detail="No content available for this entry. Fetch it first.")

    provider = entry.content.provider
    raw_path = entry.content.raw_path

    # Base response
    response = {
        "entry_id": entry_id,
        "entry_title": entry.title,
        "entry_link": entry.link,
        "provider": provider,
        "text": entry.content.text,
        "title": entry.content.title,
        "author": entry.content.author,
        "url": entry.content.url,
        "raw_path": raw_path,
    }

    # Try to load raw data if available
    if raw_path:
        raw_filename = Path(raw_path).name
        
        # Determine the correct data directory based on provider
        if provider == "exa":
            # Try new path first, then legacy path
            raw_file_path = StaticConfig.get_parse_data_dir("exa") / raw_filename
            if not raw_file_path.exists():
                raw_file_path = StaticConfig.get_exa_data_dir() / raw_filename
        elif provider == "direct":
            raw_file_path = StaticConfig.get_parse_data_dir("direct") / raw_filename
        else:
            raw_file_path = None

        if raw_file_path and raw_file_path.exists():
            try:
                with open(raw_file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                # Provider-specific raw data formatting
                if provider == "exa":
                    response["raw_data"] = {
                        "request_id": raw_data.get("requestId"),
                        "results": raw_data.get("results", []),
                        "statuses": raw_data.get("statuses", []),
                        "cost_dollars": raw_data.get("costDollars"),
                        "search_time": raw_data.get("searchTime"),
                    }
                elif provider == "direct":
                    response["raw_data"] = {
                        "url": raw_data.get("url"),
                        "final_url": raw_data.get("final_url"),
                        "fetched_at": raw_data.get("fetched_at"),
                        "status_code": raw_data.get("status_code"),
                        "content_type": raw_data.get("content_type"),
                        "html_length": len(raw_data.get("html", "")),
                    }
                else:
                    response["raw_data"] = raw_data
            except Exception as e:
                response["raw_data_error"] = str(e)

    return response


@router.get("/entries/{entry_id}/exa/preview")
async def get_exa_preview(
    request: Request,
    entry_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Get content for preview sidebar. (Deprecated: use /api/entries/{entry_id}/parse/preview)"""
    # Redirect to new endpoint
    result = await get_parse_preview(request, entry_id, db, _)
    result["deprecated"] = True
    result["message"] = "Use /api/entries/{entry_id}/parse/preview instead"
    return result


@router.post("/llm/struct/{entry_id}")
async def struct_entry_with_llm(
    request: Request,
    entry_id: int,
    force: bool = False,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """
    Manually trigger LLM structured extraction for a specific entry.
    Requires login.
    
    Args:
        entry_id: The entry ID to process.
        force: If true, overwrite existing structure. Default false.
    """
    from sqlalchemy.orm import selectinload
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.llm_struct import is_llm_enabled, extract_structured_info
    from app.models import EntryContent, EntryStructure

    config = get_runtime_config()

    if not config.llm_api_key:
        raise HTTPException(
            status_code=400,
            detail="LLM API key not configured. Update it in Settings.",
        )

    # Get the entry with its content
    result = await db.execute(
        select(Entry).where(Entry.id == entry_id).options(selectinload(Entry.content))
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    if not entry.content:
        raise HTTPException(
            status_code=400,
            detail="No Exa content available for this entry. Fetch Exa content first."
        )

    # Check if structure already exists
    existing_result = await db.execute(
        select(EntryStructure).where(EntryStructure.entry_id == entry_id)
    )
    existing = existing_result.scalar_one_or_none()
    
    if existing and not force:
        raise HTTPException(
            status_code=400,
            detail="LLM structure already exists for this entry. Use force=true to overwrite."
        )

    try:
        # Extract structured info
        struct_result = await extract_structured_info(
            title=entry.content.title,
            url=entry.content.url,
            text=entry.content.text,
        )

        if existing:
            # Update existing
            existing.model = struct_result.model
            existing.base_url = config.llm_base_url
            existing.site_type = struct_result.site_type
            existing.site_type_reason = struct_result.site_type_reason
            existing.summary = struct_result.summary
            existing.raw_json = struct_result.raw_json
            existing.status = "success"
            existing.error_message = None
        else:
            # Insert new
            stmt = (
                sqlite_insert(EntryStructure)
                .values(
                    entry_id=entry_id,
                    provider="openai_compatible",
                    model=struct_result.model,
                    base_url=config.llm_base_url,
                    site_type=struct_result.site_type,
                    site_type_reason=struct_result.site_type_reason,
                    summary=struct_result.summary,
                    raw_json=struct_result.raw_json,
                    status="success",
                )
                .on_conflict_do_nothing(index_elements=["entry_id"])
            )
            await db.execute(stmt)

        await db.commit()

        return {
            "status": "success",
            "entry_id": entry_id,
            "site_type": struct_result.site_type,
            "site_type_reason": struct_result.site_type_reason,
            "summary_length": len(struct_result.summary) if struct_result.summary else 0,
            "model": struct_result.model,
            "response_time_ms": round(struct_result.response_time_ms, 2),
            "force": force,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM extraction failed: {str(e)}")


@router.get("/entries/{entry_id}/llm/preview")
async def get_llm_preview(
    request: Request,
    entry_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Get LLM structured information for preview sidebar. Requires login."""
    from sqlalchemy.orm import selectinload
    from app.models import EntryStructure

    # Get the entry with its structure
    result = await db.execute(
        select(Entry).where(Entry.id == entry_id).options(selectinload(Entry.structured))
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    if not entry.structured:
        raise HTTPException(
            status_code=404,
            detail="No LLM structure available for this entry. Generate it first."
        )

    struct = entry.structured
    return {
        "entry_id": entry_id,
        "entry_title": entry.title,
        "site_type": struct.site_type,
        "site_type_reason": struct.site_type_reason,
        "summary": struct.summary,
        "model": struct.model,
        "provider": struct.provider,
        "status": struct.status,
        "error_message": struct.error_message,
        "created_at": struct.created_at.isoformat() if struct.created_at else None,
        "updated_at": struct.updated_at.isoformat() if struct.updated_at else None,
    }


@router.get("/status")
async def get_status(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_login_api),
):
    """Get current system status. Requires login."""
    from app.scheduler import get_next_run_time
    from app.parse import is_parse_enabled
    from sqlalchemy import func

    config = get_runtime_config()

    sub_count = await db.scalar(select(func.count(Subscription.id)))
    entry_count = await db.scalar(select(func.count(Entry.id)))

    last_run_result = await db.execute(
        select(CheckRun).order_by(desc(CheckRun.started_at)).limit(1)
    )
    last_run = last_run_result.scalar_one_or_none()

    return {
        "subscriptions": sub_count or 0,
        "entries": entry_count or 0,
        "bark_configured": bool(config.bark_device_key),
        "exa_configured": bool(config.exa_api_key),
        "parse_enabled": is_parse_enabled(),
        "parse_providers": config.parse_providers_order,
        "llm_configured": bool(config.llm_api_key),
        "next_run": get_next_run_time(),
        "last_run": {
            "id": last_run.id,
            "status": last_run.status,
            "started_at": last_run.started_at.isoformat(),
            "new_entries": last_run.total_new_entries,
        }
        if last_run
        else None,
    }
