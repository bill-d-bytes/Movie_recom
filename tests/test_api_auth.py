"""Integration tests: register, login, logout."""
import json

import pytest

from tests.conftest import login_json, register_json


@pytest.mark.integration
def test_register_201_creates_session(client):
    u, e, res = register_json(client)
    assert res.status_code == 201
    data = res.get_json()
    assert data.get("success") is True
    assert data.get("redirect") == "/discover"


@pytest.mark.integration
def test_register_400_missing_fields(client):
    res = client.post(
        "/auth/register",
        data=json.dumps({"username": "x", "email": ""}),
        content_type="application/json",
    )
    assert res.status_code == 400


@pytest.mark.integration
def test_register_409_duplicate(client):
    import uuid

    base = uuid.uuid4().hex[:10]
    email = f"shared_{base}@t.test"
    _, _, r1 = register_json(client, username=f"u1_{base}", email=email)
    assert r1.status_code == 201
    r2 = client.post(
        "/auth/register",
        data=json.dumps(
            {"username": f"u2_{base}", "email": email, "password": "UniquePass!9"}
        ),
        content_type="application/json",
    )
    assert r2.status_code == 409


@pytest.mark.integration
def test_login_401_wrong_password(client):
    u, _, r = register_json(client, password="RightPass!1")
    assert r.status_code == 201
    client.post("/auth/logout", content_type="application/json")
    bad = login_json(client, u, "WrongPass!1")
    assert bad.status_code == 401


@pytest.mark.integration
def test_login_200_after_logout(client):
    u, _, r = register_json(client)
    assert r.status_code == 201
    client.post("/auth/logout", content_type="application/json")
    ok = login_json(client, u, "TestPass!234")
    assert ok.status_code == 200
    data = ok.get_json()
    assert data.get("redirect") == "/discover"


@pytest.mark.integration
def test_logout_clears_session_for_api(client):
    register_json(client)
    assert client.get("/api/profile").status_code == 200
    client.post("/auth/logout", content_type="application/json")
    assert client.get("/api/profile").status_code == 401
