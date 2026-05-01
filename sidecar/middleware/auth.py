from fastapi import Header, HTTPException, status
from sidecar.config import settings


def get_customer_id(authorization: str | None = Header(default=None)) -> str:
    """Extract customer_id from Authorization header.

    In dev mode (SIDECAR_API_KEY unset), accepts any bearer token and uses
    the token value as the customer_id (falling back to "anonymous").
    In production mode, validates the token against SIDECAR_API_KEY and
    returns the bearer value as an opaque customer identifier.
    """
    if not settings.sidecar_api_key:
        if authorization and authorization.lower().startswith("bearer "):
            return authorization[7:] or "anonymous"
        return "anonymous"

    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization format")

    if token != settings.sidecar_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return token
