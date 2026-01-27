"""
Authentication helpers for Journal Monitor.
"""

from typing import Optional
from urllib.parse import urlencode

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

from app.config import get_runtime_config, get_auth_config
from app.config_store import verify_password


def is_logged_in(request: Request) -> bool:
    """Check if the current request has an authenticated session."""
    # Auth is always enabled now (password is set on first startup)
    return request.session.get("logged_in", False)


def login(request: Request) -> None:
    """Mark the session as logged in."""
    request.session["logged_in"] = True


def logout(request: Request) -> None:
    """Clear the session."""
    request.session.clear()


def verify_credentials(username: str, password: str) -> bool:
    """Verify login credentials against stored values."""
    runtime_config = get_runtime_config()
    auth_config = get_auth_config()
    
    # Check username
    if username != runtime_config.auth_username:
        return False
    
    # Check password hash
    return verify_password(
        password,
        auth_config.password_hash,
        auth_config.password_salt,
    )


async def require_login_api(request: Request) -> None:
    """
    FastAPI dependency that raises 401 if not logged in.
    Use: Depends(require_login_api)
    """
    if not is_logged_in(request):
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )


def require_login_redirect(request: Request, next_url: Optional[str] = None) -> Optional[RedirectResponse]:
    """
    Check if login is required and return a redirect response if so.
    Returns None if already logged in.
    """
    if is_logged_in(request):
        return None
    
    # Build redirect URL with next parameter
    if next_url is None:
        next_url = str(request.url.path)
    
    query = urlencode({"next": next_url})
    return RedirectResponse(url=f"/login?{query}", status_code=302)
