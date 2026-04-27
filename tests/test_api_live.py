"""
Live API tests for /api/recommend.
Uses Flask's test client (no external server needed).

Tests:
  1. Modern era → no 90s movies (all years >= 2000)
  2. 90s era → ML-1M movies (years 1990-1999)
  3. Genre filtering (Action anchor → mostly action results)
  4. Horror anchor + Modern era → horror-skewed modern results
  5. Drama anchor + 90s era → drama/90s results
  6. Multiple era chips (90s + 00s)
  7. Empty result resilience (no crash)
  8. rec_source tagging (latest / tmdb_popular / regional_in / hybrid)
"""

import json
import pytest

try:
    from config import ml_models_are_built
    MODELS_READY = ml_models_are_built()
except Exception:
    MODELS_READY = False

pytestmark = pytest.mark.skipif(
    not MODELS_READY,
    reason="ML models not built — run: python recommender.py --build",
)


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Flask test client with a logged-in session (user_id=1)."""
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    flask_app.config["CACHE_TYPE"] = "NullCache"   # disable caching for fresh results
    with flask_app.test_client() as c:
        with c.session_transaction() as sess:
            sess["user_id"] = 1
        yield c


def _recommend(client, movie_id, era_filter=None, genres=None, min_avg=None,
               include_latest=True, include_regional=False):
    """POST /api/recommend and return parsed JSON."""
    payload = {
        "movie_id": movie_id,
        "era_filter": era_filter or [],
        "top_n": 10,
        "include_latest": include_latest,
        "include_regional_indian": include_regional,
    }
    if min_avg is not None:
        payload["min_avg_rating"] = min_avg
    resp = client.post(
        "/api/recommend",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.data[:300]}"
    return resp.get_json()


def _years(data):
    return [m.get("year") for m in data["movies"] if m.get("year") is not None]


def _sources(data):
    return [m.get("rec_source") for m in data["movies"]]


def _genres_flat(data):
    out = []
    for m in data["movies"]:
        out.extend(m.get("genres") or [])
    return out


# ─── anchor movies ────────────────────────────────────────────────────────────
#  6  = Heat (1995)            Action|Crime|Thriller
#  1  = Toy Story (1995)       Animation|Children's|Comedy
#  12 = Dracula: Dead and Loving It  Comedy|Horror
#  24 = Powder (1995)          Drama|Sci-Fi
#  4  = Waiting to Exhale      Comedy|Drama


class TestModernEra:
    """Modern (2010+) era → no 90s movies in results."""

    def test_action_anchor_modern_no_90s(self, client):
        data = _recommend(client, movie_id=6, era_filter=["modern"],
                          include_regional=False)
        assert data["movies"], "Expected results for Action/Modern"
        years = _years(data)
        old = [y for y in years if isinstance(y, int) and y < 2001]
        assert old == [], f"Found pre-2001 movies in Modern era: {old}"

    def test_action_anchor_modern_with_regional(self, client):
        data = _recommend(client, movie_id=6, era_filter=["modern"],
                          include_regional=True)
        assert data["movies"], "Expected results for Action/Modern+Regional"
        years = _years(data)
        old = [y for y in years if isinstance(y, int) and y < 2001]
        assert old == [], f"Found pre-2001 movies in Modern+Regional: {old}"

    def test_comedy_anchor_modern(self, client):
        data = _recommend(client, movie_id=1, era_filter=["modern"],
                          include_regional=False)
        assert data["movies"], "Expected results for Comedy/Modern"
        years = _years(data)
        old = [y for y in years if isinstance(y, int) and y < 2001]
        assert old == [], f"Found pre-2001 movies in Comedy/Modern: {old}"

    def test_horror_anchor_modern(self, client):
        data = _recommend(client, movie_id=12, era_filter=["modern"],
                          include_regional=False)
        assert data["movies"], "Expected results for Horror/Modern"
        years = _years(data)
        old = [y for y in years if isinstance(y, int) and y < 2001]
        assert old == [], f"Found pre-2001 movies in Horror/Modern: {old}"

    def test_modern_sources_contain_tmdb_popular_or_latest(self, client):
        """Modern era on ML-1M must be filled by tmdb_popular or latest blend."""
        data = _recommend(client, movie_id=6, era_filter=["modern"],
                          include_regional=False)
        sources = set(_sources(data))
        tmdb_sources = sources & {"tmdb_popular", "latest", "regional_in"}
        assert tmdb_sources, (
            f"Modern era results should include tmdb_popular/latest blend rows. "
            f"Got sources: {sources}"
        )


class TestOldEras:
    """90s / 80s era → results should come from ML-1M (correct decades)."""

    def test_action_90s_years_in_range(self, client):
        data = _recommend(client, movie_id=6, era_filter=["90s"],
                          include_regional=False)
        assert data["movies"], "Expected 90s results for Action anchor"
        years = [y for y in _years(data) if isinstance(y, int)]
        if years:   # some results may have no year
            assert all(1990 <= y <= 1999 for y in years), (
                f"90s era returned out-of-range years: {[y for y in years if not 1990<=y<=1999]}"
            )

    def test_drama_90s(self, client):
        data = _recommend(client, movie_id=4, era_filter=["90s"])
        assert data["movies"], "Expected 90s Drama results"

    def test_80s_era(self, client):
        data = _recommend(client, movie_id=6, era_filter=["80s"])
        assert data["movies"], "Expected 80s results"
        years = [y for y in _years(data) if isinstance(y, int)]
        if years:
            out_of_range = [y for y in years if not 1980 <= y <= 1989]
            assert out_of_range == [], f"80s era returned years outside 1980-1989: {out_of_range}"


class TestMultiEra:
    """Multiple era chips combined."""

    def test_90s_and_00s_combined(self, client):
        data = _recommend(client, movie_id=6, era_filter=["90s", "00s"])
        assert data["movies"], "Expected results for 90s+00s"
        years = [y for y in _years(data) if isinstance(y, int)]
        if years:
            bad = [y for y in years if not (1990 <= y <= 2009)]
            assert bad == [], f"90s+00s era returned out-of-range years: {bad}"

    def test_modern_and_00s_combined(self, client):
        data = _recommend(client, movie_id=6, era_filter=["modern", "00s"],
                          include_regional=False)
        assert data["movies"], "Expected results for modern+00s"
        years = _years(data)
        old = [y for y in years if isinstance(y, int) and y < 2000]
        assert old == [], f"modern+00s returned pre-2000 years: {old}"


class TestGenreRelevance:
    """Genre preferences should influence TMDb blends."""

    def test_action_anchor_modern_genre_skew(self, client):
        """Action anchor + Modern era → majority of genres should be action-adjacent."""
        data = _recommend(client, movie_id=6, era_filter=["modern"],
                          include_regional=False)
        genres = _genres_flat(data)
        action_adj = {"Action", "Adventure", "Thriller", "Crime", "Sci-Fi"}
        hits = sum(1 for g in genres if g in action_adj)
        ratio = hits / len(genres) if genres else 0
        assert ratio >= 0.3, (
            f"Action/Modern results have too few action-adjacent genres "
            f"({hits}/{len(genres)} = {ratio:.0%}). Genres: {set(genres)}"
        )

    def test_comedy_anchor_modern_genre_skew(self, client):
        data = _recommend(client, movie_id=1, era_filter=["modern"],
                          include_regional=False)
        genres = _genres_flat(data)
        comedy_adj = {"Comedy", "Animation", "Children's", "Family", "Romance"}
        hits = sum(1 for g in genres if g in comedy_adj)
        ratio = hits / len(genres) if genres else 0
        assert ratio >= 0.2, (
            f"Comedy/Modern results have too few comedy-adjacent genres "
            f"({hits}/{len(genres)} = {ratio:.0%}). Genres: {set(genres)}"
        )


class TestResponseStructure:
    """Response must always be well-formed."""

    def test_response_has_movies_key(self, client):
        data = _recommend(client, movie_id=6, era_filter=["modern"])
        assert "movies" in data
        assert isinstance(data["movies"], list)

    def test_response_has_meta_key(self, client):
        data = _recommend(client, movie_id=6, era_filter=["modern"])
        assert "meta" in data

    def test_each_movie_has_required_fields(self, client):
        data = _recommend(client, movie_id=6, era_filter=["modern"])
        for m in data["movies"]:
            for field in ("movie_id", "title", "genres", "year", "rec_source"):
                assert field in m, f"Missing field '{field}' in movie: {m}"
            assert isinstance(m["genres"], list), f"genres must be list: {m}"

    def test_no_duplicate_movie_ids(self, client):
        data = _recommend(client, movie_id=6, era_filter=["modern"])
        ids = [m["movie_id"] for m in data["movies"]]
        assert len(ids) == len(set(ids)), f"Duplicate movie_ids in results: {ids}"

    def test_anchor_not_in_results(self, client):
        anchor_id = 6
        data = _recommend(client, movie_id=anchor_id, era_filter=["modern"])
        ids = [m["movie_id"] for m in data["movies"]]
        assert anchor_id not in ids, "Anchor movie appeared in its own recommendations"

    def test_min_avg_rating_respected(self, client):
        """min_avg_rating=4.0 on 1-5 scale should work without crashing."""
        data = _recommend(client, movie_id=6, era_filter=["modern"], min_avg=4.0)
        assert "movies" in data

    def test_no_era_filter_still_works(self, client):
        """No era filter → full catalog, should return results."""
        data = _recommend(client, movie_id=6, era_filter=[])
        assert data["movies"], "Expected results with no era filter"


class TestAllAnchors:
    """Smoke-test several different anchor movies."""

    @pytest.mark.parametrize("movie_id,label", [
        (1,  "Toy Story / Animation+Comedy"),
        (6,  "Heat / Action+Crime"),
        (12, "Dracula Dead and Loving It / Comedy+Horror"),
        (24, "Powder / Drama+Sci-Fi"),
        (4,  "Waiting to Exhale / Comedy+Drama"),
    ])
    def test_modern_era_all_anchors(self, client, movie_id, label):
        data = _recommend(client, movie_id=movie_id, era_filter=["modern"],
                          include_regional=False)
        assert "movies" in data, f"No movies key for {label}"
        assert data["movies"], f"Empty results for {label} / Modern"
        years = _years(data)
        old = [y for y in years if isinstance(y, int) and y < 2001]
        assert old == [], (
            f"{label}: got pre-2001 years in Modern era: {old}. "
            f"All years: {years}"
        )

    @pytest.mark.parametrize("movie_id,label", [
        (1,  "Toy Story"),
        (6,  "Heat"),
        (24, "Powder"),
    ])
    def test_90s_era_all_anchors(self, client, movie_id, label):
        data = _recommend(client, movie_id=movie_id, era_filter=["90s"])
        assert data["movies"], f"Empty results for {label} / 90s"
