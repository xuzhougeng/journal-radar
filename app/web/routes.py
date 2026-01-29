"""
Web page routes for Journal Monitor.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, desc, or_
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
    include_news: int = 0,
    subscription_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Recent entries page. Publicly accessible."""
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

    # Shared filters (apply to both count query and entries query)
    filters = []
    if not include_news:
        news_pattern = "%news%"
        filters.append(
            ~or_(
                Entry.title.ilike(news_pattern),
                Entry.subscription.has(Subscription.name.ilike(news_pattern)),
            )
        )
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

    query = select(Entry).options(selectinload(Entry.subscription), selectinload(Entry.content), selectinload(Entry.structured))
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
            "include_news": bool(include_news),
            "subscription_id": subscription_id_value,
            "subscriptions": subscriptions,
            "today_date": today_date,
            "parse_enabled": is_parse_enabled(),
            "llm_configured": bool(config.llm_api_key),
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
