"""
Journal Monitor - FastAPI Application Entry Point

A web application that monitors journal updates and sends push notifications.
"""

from app.logging_config import setup_logging

setup_logging()

from contextlib import asynccontextmanager
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import StaticConfig
from app.config_store import get_session_secret
from app.db import init_db, async_session


# ---------------------------------------------------------------------------
# Global Auth Middleware - protect all web pages except whitelist
# ---------------------------------------------------------------------------
# Whitelist paths that are accessible without login:
# - "/" (redirects to /entries anyway)
# - "/entries" and "/entries?..." (public listing with read-only preview)
# - "/login", "/logout"
# - "/healthz" (health check)
# - "/static/*" (assets)
# - "/api/*" (handled by per-route dependencies, mostly require_login_api)

_PUBLIC_PATH_PREFIXES = (
    "/static/",
    "/api/",
)

_PUBLIC_EXACT_PATHS = {
    "/",
    "/entries",
    "/login",
    "/logout",
    "/healthz",
}


def _is_public_path(path: str) -> bool:
    """Check if a path is publicly accessible without login."""
    # Exact matches (including query string stripped)
    base_path = path.split("?")[0]
    if base_path in _PUBLIC_EXACT_PATHS:
        return True
    # Prefix matches
    for prefix in _PUBLIC_PATH_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


class GlobalAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce authentication on all web pages.
    
    - Public paths (whitelist) are allowed without login.
    - API routes are handled by their own dependencies (return 401).
    - Other web pages redirect to /login?next=<path> if not logged in.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow public paths
        if _is_public_path(path):
            return await call_next(request)

        # Check if logged in (session is populated by SessionMiddleware)
        logged_in = False
        if "session" in request.scope:
            logged_in = request.session.get("logged_in", False)
        if logged_in:
            return await call_next(request)

        # Not logged in and not a public path -> redirect to login
        # Only redirect for non-API paths (API paths should return 401 via their dependencies)
        if not path.startswith("/api/"):
            next_url = str(request.url.path)
            if request.url.query:
                next_url += "?" + request.url.query
            query = urlencode({"next": next_url})
            return RedirectResponse(url=f"/login?{query}", status_code=302)

        # For API paths without proper auth dependency, still proceed
        # (the route's own dependency will return 401)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    StaticConfig.ensure_data_dir()
    await init_db()

    # Initialize configuration from database (or seed defaults on first run)
    from app.config_store import ensure_config

    async with async_session() as session:
        await ensure_config(session)

    # Import and start scheduler after DB and config are ready
    from app.scheduler import start_scheduler

    start_scheduler()

    yield

    # Shutdown
    from app.scheduler import shutdown_scheduler

    shutdown_scheduler()


app = FastAPI(
    title=StaticConfig.APP_NAME,
    description="Monitor journal updates and receive push notifications via Bark",
    version="0.1.0",
    lifespan=lifespan,
)

# Add session middleware with file-based secret (always enabled now)
# Set HTTPS_ONLY=false env var for plain HTTP deployments
app.add_middleware(
    SessionMiddleware,
    secret_key=get_session_secret(),
    same_site="lax",
    https_only=StaticConfig.SESSION_COOKIE_HTTPS_ONLY,
)

# Add global auth middleware (after SessionMiddleware so session is available)
# This ensures all non-public web pages require login
app.add_middleware(GlobalAuthMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/web/templates")


# Health check endpoint
@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "ok", "app": StaticConfig.APP_NAME}


# Import and include routers
from app.web.routes import router as web_router
from app.web.api import router as api_router

app.include_router(web_router)
app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=StaticConfig.DEBUG,
        log_config=None,  # keep app-controlled logging config
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
