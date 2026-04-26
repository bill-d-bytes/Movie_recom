"""
Shared pytest fixtures for CineMatch.
Uses the same SQLite + ML models as dev when present (import side effects in app).
"""
import json
import uuid

import pytest

from app import app as flask_app


@pytest.fixture
def app():
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def register_json(client, username=None, email=None, password="TestPass!234"):
    """Register via API; returns (username, email, response)."""
    u = username or f"test_{uuid.uuid4().hex[:12]}"
    e = email or f"{u}@example.test"
    res = client.post(
        "/auth/register",
        data=json.dumps({"username": u, "email": e, "password": password}),
        content_type="application/json",
    )
    return u, e, res


def login_json(client, username, password="TestPass!234"):
    return client.post(
        "/auth/login",
        data=json.dumps({"username": username, "password": password}),
        content_type="application/json",
    )
