"""Integration tests: POST /api/recommend (hybrid engine)."""
import json
import os

import pytest

from config import Config
from tests.conftest import register_json


@pytest.mark.integration
def test_recommend_400_missing_movie_id(client):
    register_json(client)
    res = client.post(
        "/api/recommend",
        data=json.dumps({"era_filter": []}),
        content_type="application/json",
    )
    assert res.status_code == 400


@pytest.mark.integration
def test_recommend_returns_list_or_service_unavailable(client):
    register_json(client)
    res = client.post(
        "/api/recommend",
        data=json.dumps({"movie_id": 1, "era_filter": ["90s"], "top_n": 10}),
        content_type="application/json",
    )
    if not os.path.exists(Config.COSINE_SIM_PATH):
        assert res.status_code == 503
        assert "error" in res.get_json()
        return
    assert res.status_code == 200
    data = res.get_json()
    movies = data["movies"] if isinstance(data, dict) and "movies" in data else data
    assert isinstance(movies, list)
    if movies:
        first = movies[0]
        assert "movie_id" in first
        assert "hybrid_score" in first
        assert "title" in first


@pytest.mark.integration
def test_recommend_uses_cache_second_call(client):
    if not os.path.exists(Config.COSINE_SIM_PATH):
        pytest.skip("ML models not built")
    register_json(client)
    body = json.dumps({"movie_id": 1, "era_filter": [], "top_n": 5})
    r1 = client.post("/api/recommend", data=body, content_type="application/json")
    r2 = client.post("/api/recommend", data=body, content_type="application/json")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.get_json() == r2.get_json()
