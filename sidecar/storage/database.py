"""aiosqlite connection management and migration runner."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiosqlite

from sidecar.config import settings

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_conn: aiosqlite.Connection | None = None
_lock = asyncio.Lock()


def _path_from_url(url: str) -> str:
    """Normalise a database URL or bare path to a filesystem path.

    Accepts both SQLAlchemy-style prefixes and bare paths::

        sqlite+aiosqlite:///./traces.db  →  ./traces.db
        sqlite:///./traces.db            →  ./traces.db
        ./traces.db                      →  ./traces.db
        :memory:                         →  :memory:
    """
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return url


async def get_db() -> aiosqlite.Connection:
    global _conn
    if _conn is None:
        raise RuntimeError("Database not initialized — call init_db() first")
    return _conn


async def init_db(db_url: str | None = None) -> None:
    """Initialise the database.  Uses *settings.database_url* by default."""
    global _conn
    async with _lock:
        if _conn is not None:
            return
        path = _path_from_url(db_url or settings.database_url)
        _conn = await aiosqlite.connect(path)
        _conn.row_factory = aiosqlite.Row
        await _conn.execute("PRAGMA journal_mode=WAL")
        await _conn.execute("PRAGMA foreign_keys=ON")
        await _bootstrap_migrations(_conn)
        await _run_migrations(_conn)
        logger.info("Database initialised at %s", path)


async def close_db() -> None:
    global _conn
    async with _lock:
        if _conn is not None:
            await _conn.close()
            _conn = None


async def _bootstrap_migrations(conn: aiosqlite.Connection) -> None:
    """Create the schema_migrations tracking table if it doesn't exist."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at REAL NOT NULL
        )
        """
    )
    await conn.commit()


async def _run_migrations(conn: aiosqlite.Connection) -> None:
    """Apply any migration files not yet recorded in schema_migrations."""
    import time

    async with conn.execute("SELECT filename FROM schema_migrations") as cur:
        applied = {row[0] for row in await cur.fetchall()}

    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        if path.name in applied:
            logger.debug("Skipping already-applied migration %s", path.name)
            continue
        logger.debug("Applying migration %s", path.name)
        sql = path.read_text()
        await conn.executescript(sql)
        await conn.execute(
            "INSERT INTO schema_migrations (filename, applied_at) VALUES (?, ?)",
            (path.name, time.time()),
        )
        await conn.commit()
