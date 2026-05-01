"""Auth middleware: extract and normalise customer identity.

Separates authentication (is this token valid?) from tenant identity
(who is this customer?).  The bearer token is hashed so the raw secret
never appears in traces or logs.
"""
from __future__ import annotations

import hashlib

from fastapi import Header, HTTPException, status

from sidecar.config import settings


def _derive_customer_id(token: str) -> str:
    """Derive an opaque, stable customer identifier from a bearer token.

    Using a hash means:
    - The raw secret is never stored in traces or logs.
    - Two requests with the same token always map to the same customer_id.
    - Customer IDs are short and URL-safe.
    """
    return "cus_" + hashlib.sha256(token.encode()).hexdigest()[:16]


def get_customer_id(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: validate the request and return an opaque customer_id.

    Dev mode (SIDECAR_API_KEY unset):
        Any bearer token is accepted.  The token is hashed to produce a
        stable customer_id.  Requests with no Authorization header get the
        special sentinel ``cus_anonymous``.

    Production mode (SIDECAR_API_KEY set):
        The bearer token must match SIDECAR_API_KEY exactly.  The hashed
        token becomes the customer_id (single-tenant).  Multi-tenant support
        will map tokens to registered tenant records in a future migration.
    """
    if not settings.sidecar_api_key:
        # Dev mode — accept anything
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            return _derive_customer_id(token) if token else "cus_anonymous"
        return "cus_anonymous"

    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization format")

    if token != settings.sidecar_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return _derive_customer_id(token)
