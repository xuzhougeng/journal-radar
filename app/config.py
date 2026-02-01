"""
Configuration management for Journal Monitor.
Runtime configuration is stored in the database and cached in-process.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """
    Runtime configuration stored in database.
    All fields have sensible defaults aligned with .env.example.
    """

    # Bark Push Notification
    bark_enabled: bool = Field(
        default=True,
        description="Enable Bark push notifications (can be disabled without clearing device key)",
    )
    bark_device_key: Optional[str] = Field(
        default=None,
        description="Bark device key for push notifications",
    )
    bark_server_url: str = Field(
        default="https://api.day.app",
        description="Bark server URL (self-hosted or default)",
    )

    # Scheduler
    check_hour: int = Field(
        default=8,
        ge=0,
        le=23,
        description="Hour of day to run scheduled checks (0-23)",
    )
    check_minute: int = Field(
        default=0,
        ge=0,
        le=59,
        description="Minute of hour to run scheduled checks (0-59)",
    )
    timezone: str = Field(
        default="Asia/Shanghai",
        description="Timezone for scheduled tasks",
    )

    # HTTP Client
    request_timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
    )
    user_agent: str = Field(
        default="JournalMonitor/1.0 (https://github.com/journal-monitor; mailto:your@email.com)",
        description="User-Agent header for HTTP requests",
    )

    # Push settings
    push_merge_entries: bool = Field(
        default=True,
        description="Merge multiple entries into one push notification per journal",
    )
    push_max_entries_per_message: int = Field(
        default=10,
        description="Maximum entries to include in a merged push message",
    )

    # Exa AI content extraction
    exa_api_key: Optional[str] = Field(
        default=None,
        description="Exa AI API key for web content extraction (x-api-key header)",
    )
    exa_livecrawl: str = Field(
        default="fallback",
        description="Exa live crawl mode for contents API (e.g. fallback/always/never)",
        pattern="^(fallback|always|never)$",
    )
    exa_livecrawl_timeout_ms: int = Field(
        default=15000,
        ge=1000,
        le=120000,
        description="Timeout in milliseconds for Exa live crawl (livecrawlTimeout)",
    )
    exa_text_max_chars: int = Field(
        default=50000,
        description="Maximum characters to store for Exa extracted text (truncate if longer)",
    )
    exa_contents_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of retries for Exa contents API on failure (0 = no retries)",
    )

    # Parse/Content fetching (multi-provider with fallback)
    parse_providers_order: list[str] = Field(
        default=["exa", "direct"],
        description="Order of providers to try for content extraction (fallback order)",
    )
    parse_min_text_chars: int = Field(
        default=200,
        ge=0,
        le=10000,
        description="Minimum text length threshold for successful extraction",
    )

    # LLM structured extraction (OpenAI-compatible API)
    llm_auto_extract: bool = Field(
        default=True,
        description="Automatically run LLM extraction on new entries during check runs",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI-compatible LLM service",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com",
        description="Base URL for OpenAI-compatible API (without /v1/chat/completions)",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name for LLM structured extraction",
    )
    llm_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout in seconds for LLM API calls",
    )
    llm_max_input_chars: int = Field(
        default=15000,
        ge=1000,
        le=100000,
        description="Maximum characters to send to LLM (truncate if longer)",
    )

    # Authentication (stored with password hash, not plaintext)
    auth_username: str = Field(
        default="admin",
        description="Username for admin login",
    )
    # Password is stored as hash with salt, not in this model directly
    # See AuthConfig for password hash storage

    class Config:
        extra = "ignore"  # Ignore extra fields when loading from DB


class AuthConfig(BaseModel):
    """
    Authentication configuration stored separately for security.
    Password is stored as PBKDF2 hash.
    """

    password_hash: str = Field(
        default="",
        description="PBKDF2 hash of admin password",
    )
    password_salt: str = Field(
        default="",
        description="Salt used for password hashing (hex encoded)",
    )
    session_secret_key: str = Field(
        default="",
        description="Secret key for signing session cookies",
    )

    class Config:
        extra = "ignore"


class StaticConfig:
    """
    Static configuration that cannot be changed at runtime.
    These are hardcoded or set via environment for initial bootstrap only.
    """

    APP_NAME: str = "Journal Monitor"
    DEBUG: bool = False
    DATABASE_URL: str = "sqlite+aiosqlite:///data/journal_monitor.sqlite3"
    DATA_DIR: Path = Path("data")

    # Session cookie security: set HTTPS_ONLY=false for plain HTTP deployments
    # Default: True (secure-by-default, requires HTTPS)
    SESSION_COOKIE_HTTPS_ONLY: bool = os.getenv("HTTPS_ONLY", "true").lower() in ("true", "1", "yes")

    @classmethod
    def get_exa_data_dir(cls) -> Path:
        """Get the Exa raw response data directory path."""
        return cls.DATA_DIR / "exa"

    @classmethod
    def get_parse_data_dir(cls, provider: str) -> Path:
        """Get the parse raw data directory for a specific provider."""
        return cls.DATA_DIR / "parse" / provider

    @classmethod
    def ensure_data_dir(cls) -> None:
        """Ensure the data directory exists."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def ensure_exa_data_dir(cls) -> None:
        """Ensure the Exa data directory exists."""
        cls.get_exa_data_dir().mkdir(parents=True, exist_ok=True)

    @classmethod
    def ensure_parse_data_dir(cls, provider: str) -> None:
        """Ensure the parse data directory for a provider exists."""
        cls.get_parse_data_dir(provider).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Runtime config cache (populated on startup from DB)
# =============================================================================

_runtime_config: Optional[RuntimeConfig] = None
_auth_config: Optional[AuthConfig] = None


def get_runtime_config() -> RuntimeConfig:
    """
    Get the cached runtime configuration.
    Must be initialized on startup via config_store.ensure_config().
    """
    global _runtime_config
    if _runtime_config is None:
        # Return defaults if not yet initialized (should not happen in normal flow)
        _runtime_config = RuntimeConfig()
    return _runtime_config


def get_auth_config() -> AuthConfig:
    """
    Get the cached auth configuration.
    Must be initialized on startup via config_store.ensure_config().
    """
    global _auth_config
    if _auth_config is None:
        _auth_config = AuthConfig()
    return _auth_config


def set_runtime_config(config: RuntimeConfig) -> None:
    """Update the cached runtime configuration."""
    global _runtime_config
    _runtime_config = config


def set_auth_config(config: AuthConfig) -> None:
    """Update the cached auth configuration."""
    global _auth_config
    _auth_config = config


# =============================================================================
# Legacy compatibility shim (for gradual migration)
# =============================================================================

class _LegacySettingsShim:
    """
    Shim to provide backward compatibility during migration.
    Reads from runtime config cache.
    """

    @property
    def app_name(self) -> str:
        return StaticConfig.APP_NAME

    @property
    def debug(self) -> bool:
        return StaticConfig.DEBUG

    @property
    def database_url(self) -> str:
        return StaticConfig.DATABASE_URL

    @property
    def bark_enabled(self) -> bool:
        return get_runtime_config().bark_enabled

    @property
    def bark_device_key(self) -> Optional[str]:
        return get_runtime_config().bark_device_key

    @property
    def bark_server_url(self) -> str:
        return get_runtime_config().bark_server_url

    @property
    def check_hour(self) -> int:
        return get_runtime_config().check_hour

    @property
    def check_minute(self) -> int:
        return get_runtime_config().check_minute

    @property
    def timezone(self) -> str:
        return get_runtime_config().timezone

    @property
    def request_timeout(self) -> int:
        return get_runtime_config().request_timeout

    @property
    def user_agent(self) -> str:
        return get_runtime_config().user_agent

    @property
    def push_merge_entries(self) -> bool:
        return get_runtime_config().push_merge_entries

    @property
    def push_max_entries_per_message(self) -> int:
        return get_runtime_config().push_max_entries_per_message

    @property
    def exa_api_key(self) -> Optional[str]:
        return get_runtime_config().exa_api_key

    @property
    def exa_text_max_chars(self) -> int:
        return get_runtime_config().exa_text_max_chars

    @property
    def parse_providers_order(self) -> list[str]:
        return get_runtime_config().parse_providers_order

    @property
    def parse_min_text_chars(self) -> int:
        return get_runtime_config().parse_min_text_chars

    @property
    def llm_auto_extract(self) -> bool:
        return get_runtime_config().llm_auto_extract

    @property
    def llm_api_key(self) -> Optional[str]:
        return get_runtime_config().llm_api_key

    @property
    def llm_base_url(self) -> str:
        return get_runtime_config().llm_base_url

    @property
    def llm_model(self) -> str:
        return get_runtime_config().llm_model

    @property
    def llm_timeout(self) -> int:
        return get_runtime_config().llm_timeout

    @property
    def llm_max_input_chars(self) -> int:
        return get_runtime_config().llm_max_input_chars

    @property
    def auth_username(self) -> str:
        return get_runtime_config().auth_username

    @property
    def auth_password(self) -> Optional[str]:
        # For legacy compatibility, return None (password is now hashed in DB)
        return None

    @property
    def session_secret_key(self) -> Optional[str]:
        return get_auth_config().session_secret_key or None

    @property
    def auth_enabled(self) -> bool:
        """Auth is always enabled now (password is set on first startup)."""
        auth = get_auth_config()
        return bool(auth.password_hash and auth.session_secret_key)

    @property
    def data_dir(self) -> Path:
        return StaticConfig.DATA_DIR

    @property
    def exa_data_dir(self) -> Path:
        return StaticConfig.get_exa_data_dir()

    def ensure_data_dir(self) -> None:
        StaticConfig.ensure_data_dir()

    def ensure_exa_data_dir(self) -> None:
        StaticConfig.ensure_exa_data_dir()


# Global settings instance (legacy shim for backward compatibility)
settings = _LegacySettingsShim()
