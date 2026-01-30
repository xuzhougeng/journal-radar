"""
Web page routes for Journal Monitor.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request, Depends, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, desc, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_db
from app.models import Subscription, Entry, CheckRun
from app.config import get_runtime_config
from app.scheduler import get_next_run_time
from app.web.auth import is_logged_in, login, logout, verify_credentials, require_login_redirect

router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")


@router.get("/", response_class=HTMLResponse)
async def home():
    """Home page redirects to entries."""
    return RedirectResponse(url="/entries", status_code=302)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    """Dashboard page. Redirects to /login if not logged in."""
    # If not logged in, redirect to login page
    if not is_logged_in(request):
        return RedirectResponse(url="/login?next=/dashboard", status_code=302)

    config = get_runtime_config()

    # Get subscription count
    sub_count = await db.scalar(select(func.count(Subscription.id)))

    # Get recent entries count (last 7 days)
    entry_count = await db.scalar(select(func.count(Entry.id)))

    # Get last check run
    last_run_result = await db.execute(
        select(CheckRun).order_by(desc(CheckRun.started_at)).limit(1)
    )
    last_run = last_run_result.scalar_one_or_none()

    # Get next scheduled run
    next_run = get_next_run_time()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "settings": config,
            "subscription_count": sub_count or 0,
            "entry_count": entry_count or 0,
            "last_run": last_run,
            "next_run": next_run,
            "logged_in": True,
        },
    )


@router.get("/subscriptions", response_class=HTMLResponse)
async def subscriptions_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Subscriptions management page. Requires login."""
    redirect = require_login_redirect(request)
    if redirect:
        return redirect

    result = await db.execute(select(Subscription).order_by(desc(Subscription.created_at)))
    subscriptions = result.scalars().all()

    return templates.TemplateResponse(
        "subscriptions.html",
        {
            "request": request,
            "subscriptions": subscriptions,
            "logged_in": True,
        },
    )


@router.get("/entries", response_class=HTMLResponse)
async def entries_page(
    request: Request,
    sort: str = "published",
    page: int = 1,
    page_size: int = 10,
    type_filter: list[str] | None = Query(None),
    tag_filter: list[str] | None = Query(None),
    subscription_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Recent entries page. Publicly accessible."""
    from app.entry_type import VALID_ARTICLE_TYPES
    from app.models import EntryType, EntryStructure, Tag
    
    config = get_runtime_config()
    today_date = datetime.now(ZoneInfo(config.timezone)).date()
    if page < 1:
        page = 1
    # Keep it sane for UI / DB load
    if page_size < 1:
        page_size = 1
    if page_size > 200:
        page_size = 200

    subscription_id_value = None
    if subscription_id:
        try:
            subscription_id_value = int(subscription_id)
        except ValueError:
            subscription_id_value = None

    # Parse type_filter from query params (can be multiple values)
    type_filter_values = []
    if type_filter:
        # Handle both single value and list
        if isinstance(type_filter, str):
            type_filter_values = [type_filter] if type_filter in VALID_ARTICLE_TYPES else []
        else:
            type_filter_values = [t for t in type_filter if t in VALID_ARTICLE_TYPES]

    # Parse tag_filter from query params (can be multiple values)
    tag_filter_values = []
    if tag_filter:
        if isinstance(tag_filter, str):
            tag_filter_values = [tag_filter.strip().lower()] if tag_filter.strip() else []
        else:
            tag_filter_values = [t.strip().lower() for t in tag_filter if t.strip()]

    # Shared filters (apply to both count query and entries query)
    filters = []
    if type_filter_values:
        # Filter by effective type, falling back to structured type if no EntryType exists.
        type_filter_set = set(type_filter_values)
        fallback_structured_types = [t for t in type_filter_values if t != "other"]
        type_conditions = [
            Entry.type_info.has(EntryType.effective_type.in_(type_filter_values))
        ]
        if fallback_structured_types:
            type_conditions.append(
                and_(
                    ~Entry.type_info.has(),
                    Entry.structured.has(
                        EntryStructure.site_type.in_(fallback_structured_types)
                    ),
                )
            )
        if "other" in type_filter_set:
            type_conditions.append(
                and_(~Entry.type_info.has(), ~Entry.structured.has())
            )
            type_conditions.append(
                and_(
                    ~Entry.type_info.has(),
                    Entry.structured.has(EntryStructure.site_type == "other"),
                )
            )
        filters.append(or_(*type_conditions))
    if tag_filter_values:
        # Filter by tag key (OR logic: entry has ANY of the specified tags)
        filters.append(Entry.tags.any(Tag.key.in_(tag_filter_values)))
    if subscription_id_value:
        filters.append(Entry.subscription_id == subscription_id_value)

    total_entries_query = select(func.count(Entry.id))
    if filters:
        total_entries_query = total_entries_query.where(*filters)
    total_entries = await db.scalar(total_entries_query) or 0
    total_pages = max(1, (total_entries + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages

    offset = (page - 1) * page_size

    query = select(Entry).options(selectinload(Entry.subscription), selectinload(Entry.content), selectinload(Entry.structured), selectinload(Entry.type_info), selectinload(Entry.tags))
    if filters:
        query = query.where(*filters)
    if sort == "journal":
        # NULL journal_name last, then alphabetical, then newest first within journal
        query = (
            query.join(Subscription)
            .order_by(
                Subscription.name,
            desc(Entry.published_at),
            desc(Entry.fetched_at),
            )
        )
    elif sort == "published":
        # Newest published first; fall back to fetched time for tie-breaks
        query = query.order_by(desc(Entry.published_at), desc(Entry.fetched_at))
    else:
        # Default: newest published first
        sort = "published"
        query = query.order_by(desc(Entry.published_at), desc(Entry.fetched_at))

    result = await db.execute(query.limit(page_size).offset(offset))
    entries = result.scalars().all()

    subscriptions_result = await db.execute(select(Subscription).order_by(Subscription.name))
    subscriptions = subscriptions_result.scalars().all()

    from app.parse import is_parse_enabled

    return templates.TemplateResponse(
        "entries.html",
        {
            "request": request,
            "entries": entries,
            "sort": sort,
            "page": page,
            "page_size": page_size,
            "total_entries": total_entries,
            "total_pages": total_pages,
            "type_filter": type_filter_values,
            "tag_filter": tag_filter_values,
            "valid_types": sorted(VALID_ARTICLE_TYPES),
            "subscription_id": subscription_id_value,
            "subscriptions": subscriptions,
            "today_date": today_date,
            "parse_enabled": is_parse_enabled(),
            "llm_configured": bool(config.llm_api_key),
            "logged_in": is_logged_in(request),
        },
    )


@router.get("/tags", response_class=HTMLResponse)
async def tags_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Tags browsing page. Requires login."""
    redirect = require_login_redirect(request)
    if redirect:
        return redirect

    from app.models import Tag, entry_tags

    # Query all tags with entry counts
    query = (
        select(Tag, func.count(entry_tags.c.entry_id).label("entry_count"))
        .outerjoin(entry_tags, Tag.id == entry_tags.c.tag_id)
        .group_by(Tag.id)
        .order_by(desc(func.count(entry_tags.c.entry_id)), Tag.name)
    )
    result = await db.execute(query)
    tags_with_counts = [
        {"tag": tag, "entry_count": entry_count}
        for tag, entry_count in result.all()
    ]

    return templates.TemplateResponse(
        "tags.html",
        {
            "request": request,
            "tags": tags_with_counts,
            "logged_in": is_logged_in(request),
        },
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page. Requires login."""
    redirect = require_login_redirect(request)
    if redirect:
        return redirect

    config = get_runtime_config()

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "settings": config,
            "bark_configured": bool(config.bark_device_key),
            "exa_configured": bool(config.exa_api_key),
            "llm_configured": bool(config.llm_api_key),
            "logged_in": True,
        },
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/entries"):
    """Login page."""
    # If already logged in, redirect to next
    if is_logged_in(request):
        return RedirectResponse(url=next, status_code=302)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "next": next,
            "error": None,
            "logged_in": False,
        },
    )


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/entries"),
):
    """Handle login form submission."""
    if verify_credentials(username, password):
        login(request)
        return RedirectResponse(url=next, status_code=302)

    # Invalid credentials
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "next": next,
            "error": "Invalid username or password",
            "logged_in": False,
        },
        # Important: keep this as 200 so browsers render the login page with an
        # inline error message, instead of treating it like an auth challenge.
        status_code=200,
    )


@router.get("/logout")
async def logout_route(request: Request):
    """Logout and redirect to login page."""
    logout(request)
    return RedirectResponse(url="/login", status_code=302)
