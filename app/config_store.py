"""
Configuration store for Journal Monitor.
Handles DB persistence of runtime configuration with in-process caching.
"""

import hashlib
import json
import logging
import secrets
import string
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    RuntimeConfig,
    AuthConfig,
    StaticConfig,
    set_runtime_config,
    set_auth_config,
    get_runtime_config,
    get_auth_config,
)
from app.models import AppConfig

logger = logging.getLogger(__name__)

# Password hashing parameters
PBKDF2_ITERATIONS = 600_000  # OWASP recommended minimum for PBKDF2-SHA256
PBKDF2_HASH_NAME = "sha256"
SALT_BYTES = 32

# Session secret file path (stored separately for sync access at startup)
SESSION_SECRET_FILE = StaticConfig.DATA_DIR / ".session_secret"


def generate_random_password(length: int = 16) -> str:
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_session_secret() -> str:
    """Generate a secure session secret key."""
    return secrets.token_hex(32)


def ensure_session_secret_file() -> str:
    """
    Ensure session secret file exists, creating it if necessary.
    This is called synchronously at module load time.
    
    Returns:
        The session secret.
    """
    StaticConfig.ensure_data_dir()
    
    if SESSION_SECRET_FILE.exists():
        try:
            secret = SESSION_SECRET_FILE.read_text().strip()
            if len(secret) >= 32:
                return secret
        except Exception:
            pass
    
    # Generate new secret
    secret = generate_session_secret()
    try:
        SESSION_SECRET_FILE.write_text(secret)
        SESSION_SECRET_FILE.chmod(0o600)  # Restrict permissions
        logger.info("Generated new session secret")
    except Exception as e:
        logger.warning(f"Could not write session secret file: {e}")
    
    return secret


# Load session secret at module import time (synchronous)
_session_secret: str = ensure_session_secret_file()


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Hash a password using PBKDF2-SHA256.
    
    Args:
        password: The plaintext password.
        salt: Optional salt bytes. If None, generates new random salt.
    
    Returns:
        Tuple of (password_hash_hex, salt_hex)
    """
    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)
    
    password_hash = hashlib.pbkdf2_hmac(
        PBKDF2_HASH_NAME,
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return password_hash.hex(), salt.hex()


def verify_password(password: str, password_hash_hex: str, salt_hex: str) -> bool:
    """
    Verify a password against a stored hash.
    
    Args:
        password: The plaintext password to verify.
        password_hash_hex: The stored password hash (hex encoded).
        salt_hex: The stored salt (hex encoded).
    
    Returns:
        True if password matches, False otherwise.
    """
    if not password_hash_hex or not salt_hex:
        return False
    
    try:
        salt = bytes.fromhex(salt_hex)
        computed_hash = hashlib.pbkdf2_hmac(
            PBKDF2_HASH_NAME,
            password.encode("utf-8"),
            salt,
            PBKDF2_ITERATIONS,
        )
        return secrets.compare_digest(computed_hash.hex(), password_hash_hex)
    except (ValueError, TypeError):
        return False


async def ensure_config(session: AsyncSession) -> Optional[str]:
    """
    Ensure configuration exists in database.
    If no config exists, creates default config with random admin password.
    
    Args:
        session: Database session.
    
    Returns:
        The generated admin password (only on first initialization), or None if config already existed.
    """
    result = await session.execute(select(AppConfig).where(AppConfig.id == 1))
    existing = result.scalar_one_or_none()
    
    generated_password = None
    
    if existing is None:
        # First time initialization
        logger.info("Initializing application configuration...")
        
        # Generate random admin password
        generated_password = generate_random_password(16)
        password_hash, password_salt = hash_password(generated_password)
        
        # Create default runtime config
        runtime_config = RuntimeConfig()
        
        # Create auth config with generated credentials
        # Session secret is stored in file, not DB
        auth_config = AuthConfig(
            password_hash=password_hash,
            password_salt=password_salt,
            session_secret_key=_session_secret,  # Use file-based secret
        )
        
        # Save to database
        app_config = AppConfig(
            id=1,
            runtime_json=runtime_config.model_dump_json(),
            auth_json=auth_config.model_dump_json(),
        )
        session.add(app_config)
        await session.commit()
        
        # Update in-process cache
        set_runtime_config(runtime_config)
        set_auth_config(auth_config)
        
        logger.info("=" * 60)
        logger.info("FIRST TIME SETUP - ADMIN CREDENTIALS")
        logger.info("=" * 60)
        logger.info(f"Username: {runtime_config.auth_username}")
        logger.info(f"Password: {generated_password}")
        logger.info("=" * 60)
        logger.info("Please save this password! It will not be shown again.")
        logger.info("You can change it later in Settings > Reset Password.")
        logger.info("=" * 60)
    else:
        # Load existing config into cache
        await load_config_to_cache(session)
    
    return generated_password


async def load_config_to_cache(session: AsyncSession) -> None:
    """
    Load configuration from database into in-process cache.
    
    Args:
        session: Database session.
    """
    result = await session.execute(select(AppConfig).where(AppConfig.id == 1))
    app_config = result.scalar_one_or_none()
    
    if app_config is None:
        logger.warning("No configuration found in database, using defaults")
        set_runtime_config(RuntimeConfig())
        set_auth_config(AuthConfig())
        return
    
    # Parse runtime config
    try:
        runtime_data = json.loads(app_config.runtime_json)
        runtime_config = RuntimeConfig.model_validate(runtime_data)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to parse runtime config: {e}, using defaults")
        runtime_config = RuntimeConfig()
    
    # Parse auth config
    try:
        auth_data = json.loads(app_config.auth_json)
        auth_config = AuthConfig.model_validate(auth_data)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to parse auth config: {e}, using defaults")
        auth_config = AuthConfig()
    
    set_runtime_config(runtime_config)
    set_auth_config(auth_config)
    
    logger.debug("Configuration loaded from database")


async def get_db_config(session: AsyncSession) -> Optional[AppConfig]:
    """
    Get the raw AppConfig model from database.
    
    Args:
        session: Database session.
    
    Returns:
        AppConfig model or None.
    """
    result = await session.execute(select(AppConfig).where(AppConfig.id == 1))
    return result.scalar_one_or_none()


async def update_runtime_config(
    session: AsyncSession,
    updates: dict,
) -> RuntimeConfig:
    """
    Update runtime configuration in database and cache.
    
    Args:
        session: Database session.
        updates: Dictionary of field updates.
    
    Returns:
        Updated RuntimeConfig.
    """
    result = await session.execute(select(AppConfig).where(AppConfig.id == 1))
    app_config = result.scalar_one_or_none()
    
    if app_config is None:
        raise ValueError("Configuration not initialized")
    
    # Load current config
    try:
        current_data = json.loads(app_config.runtime_json)
    except json.JSONDecodeError:
        current_data = {}
    
    # Apply updates
    current_data.update(updates)
    
    # Validate with Pydantic model
    new_config = RuntimeConfig.model_validate(current_data)
    
    # Save to database
    app_config.runtime_json = new_config.model_dump_json()
    await session.commit()
    
    # Update cache
    set_runtime_config(new_config)
    
    logger.info(f"Runtime configuration updated: {list(updates.keys())}")
    return new_config


async def update_auth_username(
    session: AsyncSession,
    new_username: str,
) -> None:
    """
    Update the admin username.
    
    Args:
        session: Database session.
        new_username: New username.
    """
    # Username is stored in runtime config
    await update_runtime_config(session, {"auth_username": new_username})


async def update_auth_password(
    session: AsyncSession,
    new_password: str,
) -> None:
    """
    Update the admin password (stores hash).
    
    Args:
        session: Database session.
        new_password: New plaintext password.
    """
    result = await session.execute(select(AppConfig).where(AppConfig.id == 1))
    app_config = result.scalar_one_or_none()
    
    if app_config is None:
        raise ValueError("Configuration not initialized")
    
    # Hash new password
    password_hash, password_salt = hash_password(new_password)
    
    # Load current auth config
    try:
        auth_data = json.loads(app_config.auth_json)
    except json.JSONDecodeError:
        auth_data = {}
    
    # Update password fields
    auth_data["password_hash"] = password_hash
    auth_data["password_salt"] = password_salt
    
    # Validate and save
    new_auth = AuthConfig.model_validate(auth_data)
    app_config.auth_json = new_auth.model_dump_json()
    await session.commit()
    
    # Update cache
    set_auth_config(new_auth)
    
    logger.info("Admin password updated")


async def rotate_admin_password(session: AsyncSession) -> str:
    """
    Generate a new random admin password.
    
    Args:
        session: Database session.
    
    Returns:
        The new generated password.
    """
    new_password = generate_random_password(16)
    await update_auth_password(session, new_password)
    return new_password


async def regenerate_session_secret(session: AsyncSession) -> None:
    """
    Regenerate the session secret key.
    This will invalidate all existing sessions.
    
    Args:
        session: Database session.
    """
    result = await session.execute(select(AppConfig).where(AppConfig.id == 1))
    app_config = result.scalar_one_or_none()
    
    if app_config is None:
        raise ValueError("Configuration not initialized")
    
    # Load current auth config
    try:
        auth_data = json.loads(app_config.auth_json)
    except json.JSONDecodeError:
        auth_data = {}
    
    # Generate new session secret
    auth_data["session_secret_key"] = generate_session_secret()
    
    # Validate and save
    new_auth = AuthConfig.model_validate(auth_data)
    app_config.auth_json = new_auth.model_dump_json()
    await session.commit()
    
    # Update cache
    set_auth_config(new_auth)
    
    logger.info("Session secret regenerated - all existing sessions invalidated")


def get_session_secret() -> str:
    """
    Get the session secret key from file.
    Used by SessionMiddleware at module load time.
    
    Returns:
        Session secret key.
    """
    return _session_secret
