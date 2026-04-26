"""Unit tests: password hashing (auth crypto)."""
import pytest
from werkzeug.security import check_password_hash, generate_password_hash


@pytest.mark.unit
def test_password_hash_round_trip():
    h = generate_password_hash("secret-value", method="pbkdf2:sha256")
    assert check_password_hash(h, "secret-value")
    assert not check_password_hash(h, "wrong")
