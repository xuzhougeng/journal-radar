"""
Database setup and session management for Journal Monitor.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import StaticConfig


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""

    pass


# Create async engine using static configuration
engine = create_async_engine(
    StaticConfig.DATABASE_URL,
    echo=StaticConfig.DEBUG,
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """Dependency for getting database sessions."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize the database, creating all tables."""
    # Import models to register them with Base
    from app import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
