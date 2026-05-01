"""aiosqlite connection management and migration runner."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_conn: aiosqlite.Connection | None = None
_lock = asyncio.Lock()


async def get_db() -> aiosqlite.Connection:
    global _conn
    if _conn is None:
        raise RuntimeError("Database not initialized — call init_db() first")
    return _conn


async def init_db(db_path: str = "./traces.db") -> None:
    global _conn
    async with _lock:
        if _conn is not None:
            return
        _conn = await aiosqlite.connect(db_path)
        _conn.row_factory = aiosqlite.Row
        await _conn.execute("PRAGMA journal_mode=WAL")
        await _conn.execute("PRAGMA foreign_keys=ON")
        await _run_migrations(_conn)
        logger.info("Database initialized at %s", db_path)


async def close_db() -> None:
    global _conn
    async with _lock:
        if _conn is not None:
            await _conn.close()
            _conn = None


async def _run_migrations(conn: aiosqlite.Connection) -> None:
    migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    for path in migration_files:
        logger.debug("Applying migration %s", path.name)
        sql = path.read_text()
        await conn.executescript(sql)
    await conn.commit()
