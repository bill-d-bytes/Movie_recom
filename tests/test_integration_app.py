"""Integration tests: public pages, template modes, protected routes."""
import pytest

from tests.conftest import register_json


@pytest.mark.integration
def test_login_page_200_unauthenticated(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Enter the lobby" in res.data


@pytest.mark.integration
def test_register_page_200_shows_signup(client):
    res = client.get("/auth/register")
    assert res.status_code == 200
    assert b"Create your account" in res.data
    assert b"reg-email" in res.data or b' id="reg-email"' in res.data


@pytest.mark.integration
def test_discover_redirects_unauthenticated(client):
    res = client.get("/discover", follow_redirects=False)
    assert res.status_code == 302
    # Must not keep user on the discover URL
    assert "discover" not in (res.location or "").lower()


@pytest.mark.integration
def test_protected_pages_200_with_session(client):
    register_json(client)
    for path in ("/discover", "/profile", "/preferences", "/filters", "/movie/1"):
        assert client.get(path).status_code == 200
