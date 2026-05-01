"""Shared pytest fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio
import respx
import httpx

# ── App / DB fixtures ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db(tmp_path):
    """Isolated in-memory-like SQLite DB for each test."""
    import sidecar.storage.database as db_module

    db_path = str(tmp_path / "test.db")
    # Reset singleton
    db_module._conn = None
    await db_module.init_db(db_path)
    yield db_module._conn
    await db_module.close_db()


@pytest_asyncio.fixture
async def client(db):
    """AsyncClient wired to the FastAPI app with test DB already initialized."""
    from sidecar.main import app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Fixtures data ──────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def openai_response_json() -> dict:
    return json.loads((FIXTURES_DIR / "openai_response.json").read_text())


@pytest.fixture
def stream_bytes() -> bytes:
    return (FIXTURES_DIR / "openai_stream_chunks.txt").read_bytes()


# ── respx mock router ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_openai(openai_response_json):
    """Mock the OpenAI upstream endpoint."""
    with respx.mock(base_url="https://api.openai.com", assert_all_called=False) as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_response_json)
        )
        yield mock


@pytest.fixture
def mock_openai_stream(stream_bytes):
    """Mock the OpenAI upstream endpoint with a streaming response."""
    with respx.mock(base_url="https://api.openai.com", assert_all_called=False) as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=stream_bytes,
                headers={"content-type": "text/event-stream"},
            )
        )
        yield mock
