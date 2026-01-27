"""
Journal Monitor - FastAPI Application Entry Point

A web application that monitors journal updates and sends push notifications.
"""

from app.logging_config import setup_logging

setup_logging()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.config import StaticConfig
from app.config_store import get_session_secret
from app.db import init_db, async_session


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
app.add_middleware(
    SessionMiddleware,
    secret_key=get_session_secret(),
    same_site="lax",
    https_only=not StaticConfig.DEBUG,
)

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
