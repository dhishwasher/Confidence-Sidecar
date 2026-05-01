"""Unit tests for customer ID derivation."""
import hashlib

from sidecar.middleware.auth import _derive_customer_id


def test_customer_id_is_hashed():
    cid = _derive_customer_id("my-secret-token")
    assert cid.startswith("cus_")
    assert "my-secret-token" not in cid


def test_customer_id_is_deterministic():
    assert _derive_customer_id("abc") == _derive_customer_id("abc")


def test_different_tokens_give_different_ids():
    assert _derive_customer_id("token-a") != _derive_customer_id("token-b")


def test_customer_id_length():
    cid = _derive_customer_id("any-token")
    # "cus_" + 16 hex chars = 20 chars
    assert len(cid) == 20
