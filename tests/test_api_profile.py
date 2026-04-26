"""Integration tests: /api/profile."""
import json
import uuid

import pytest

from tests.conftest import register_json


@pytest.mark.integration
def test_profile_401_unauthenticated(client):
    assert client.get("/api/profile").status_code == 401


@pytest.mark.integration
def test_profile_get_returns_user(client):
    u, e, r = register_json(client)
    assert r.status_code == 201
    res = client.get("/api/profile")
    assert res.status_code == 200
    data = res.get_json()
    assert data["username"] == u
    assert data["email"] == e
    assert isinstance(data["preferred_genres"], list)


@pytest.mark.integration
def test_profile_put_updates_email(client):
    register_json(client)
    new_email = f"new_{uuid.uuid4().hex}@t.test"
    res = client.put(
        "/api/profile",
        data=json.dumps({"email": new_email}),
        content_type="application/json",
    )
    assert res.status_code == 200
    again = client.get("/api/profile").get_json()
    assert again["email"] == new_email
