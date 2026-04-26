"""Integration tests: movie search, trending, detail."""
import json
from unittest.mock import patch

import pytest

from tests.conftest import register_json


@pytest.mark.integration
def test_movies_401_unauthenticated(client):
    assert client.get("/api/movies/search?q=The").status_code == 401


@pytest.mark.integration
def test_movies_search_short_query_empty(client):
    register_json(client)
    res = client.get("/api/movies/search?q=a")
    assert res.status_code == 200
    assert res.get_json() == []


@pytest.mark.integration
def test_movies_search_finds_titles(client):
    register_json(client)
    res = client.get("/api/movies/search?q=Toy")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)
    if data:
        assert "movie_id" in data[0]
        assert "title" in data[0]


@pytest.mark.integration
def test_movies_suggest_shape(client):
    register_json(client)
    res = client.get("/api/movies/suggest?q=Star")
    assert res.status_code == 200
    data = res.get_json()
    assert "results" in data
    assert isinstance(data["results"], list)
    if data["results"]:
        row = data["results"][0]
        assert "source" in row
        assert "title" in row
        assert "in_catalog" in row


@pytest.mark.integration
def test_movies_trending(client):
    register_json(client)
    res = client.get("/api/movies/trending")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)


@pytest.mark.integration
def test_movie_detail_200_and_404(client):
    register_json(client)
    ok = client.get("/api/movies/1")
    assert ok.status_code == 200
    m = ok.get_json()
    assert m.get("movie_id") == 1
    assert isinstance(m.get("genres"), list)
    miss = client.get("/api/movies/99999999")
    assert miss.status_code == 404


@pytest.mark.integration
def test_modern_tmdb_401_unauthenticated(client):
    assert client.get("/api/movies/modern-tmdb").status_code == 401


@pytest.mark.integration
@patch("app.discover_tmdb_movies_modern")
def test_modern_tmdb_200_shape(mock_disc, client):
    register_json(client)
    mock_disc.return_value = (
        [
            {
                "tmdb_id": 555000,
                "title": "Example Modern",
                "year": 2022,
                "poster_url": "https://example.test/p.jpg",
            },
        ],
        99,
        None,
    )
    r = client.get("/api/movies/modern-tmdb?page=1")
    assert r.status_code == 200
    j = r.get_json()
    assert j.get("source") == "tmdb_discover"
    assert j.get("total_pages") == 99
    assert len(j.get("results") or []) == 1
    assert j["results"][0].get("tmdb_id") == 555000
    assert "in_app_movie_id" in j["results"][0]


@pytest.mark.integration
def test_tmdb_related_401_unauthenticated(client):
    assert client.get("/api/movies/tmdb-related?movie_id=1").status_code == 401


@pytest.mark.integration
def test_tmdb_related_400(client):
    register_json(client)
    assert client.get("/api/movies/tmdb-related").status_code == 400


@pytest.mark.integration
@patch("app.fetch_tmdb_similar_and_recommendations")
@patch("app.resolve_tmdb_id_for_cinematch_row")
def test_tmdb_related_200_shape(mock_resolve, mock_fetch, client):
    register_json(client)
    mock_resolve.return_value = 55
    mock_fetch.return_value = (
        [
            {
                "tmdb_id": 900001,
                "title": "Zzyzx Unique Title Xq9",
                "year": 2001,
                "poster_url": "https://example.test/p.png",
            },
        ],
        None,
    )
    res = client.get("/api/movies/tmdb-related?movie_id=1")
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("source") == "tmdb"
    assert data.get("tmdb_id") == 55
    assert len(data.get("results") or []) == 1
    r0 = data["results"][0]
    assert r0.get("tmdb_id") == 900001
    assert "in_app_movie_id" in r0


@pytest.mark.integration
@patch("app.resolve_tmdb_id_for_cinematch_row")
def test_tmdb_related_unresolved_tmdb_graceful(mock_resolve, client):
    register_json(client)
    mock_resolve.return_value = None
    res = client.get("/api/movies/tmdb-related?movie_id=1")
    assert res.status_code == 200
    j = res.get_json()
    assert j.get("tmdb_id") is None
    assert j.get("results") == []
    assert "message" in j
