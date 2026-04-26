"""Integration tests: /api/preferences."""
import json

import pytest

from tests.conftest import register_json


@pytest.mark.integration
def test_preferences_401_unauthenticated(client):
    assert client.get("/api/preferences").status_code == 401


@pytest.mark.integration
def test_preferences_get_put(client):
    register_json(client)
    g = client.get("/api/preferences")
    assert g.status_code == 200
    r = client.put(
        "/api/preferences",
        data=json.dumps(
            {"preferred_genres": ["Action", "Drama"], "age": 25, "gender": "M"}
        ),
        content_type="application/json",
    )
    assert r.status_code == 200
    g2 = client.get("/api/preferences").get_json()
    assert "Action" in g2["preferred_genres"]
    assert g2["age"] == 25
    assert g2["gender"] == "M"
