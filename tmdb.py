"""
tmdb.py — TMDB API integration for fetching poster URLs and overviews.
Caches results in the DB to avoid repeat API calls.
"""
from __future__ import annotations

import re
import time
from datetime import date
import requests
from config import Config
from database import get_db

# Brief pause between TMDB calls when trying many fallbacks (rate limits)
_TMDB_BETWEEN_ATTEMPTS_S = 0.05

# Transient network drops (e.g. ConnectionResetError) — retry with backoff
_TMDB_MAX_ATTEMPTS = 4
_TMDB_RETRY_BACKOFF_S = (0.35, 0.9, 1.8)


def _tmdb_get(url: str, params: dict = None, timeout: int = 12):
    """
    GET against TMDb with retries. Handles connection reset / timeout / chunked errors
    that are common on flaky networks or when the server closes an idle connection.
    """
    params = params or {}
    last_exc: Exception | None = None
    for attempt in range(_TMDB_MAX_ATTEMPTS):
        if attempt:
            b = _TMDB_RETRY_BACKOFF_S[min(attempt - 1, len(_TMDB_RETRY_BACKOFF_S) - 1)]
            time.sleep(b)
        try:
            return requests.get(url, params=params, timeout=timeout)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last_exc = e
            continue
    if last_exc is not None:
        raise last_exc
    raise requests.RequestException("TMDb request failed with no prior exception")


def _ml_title_to_search_queries(title: str):
    """
    Yields 1+ query strings for TMDB, from the ml-1m style title.
    Strips (YYYY) and optional (a.k.a. …); adds variants so edge cases match TMDb.
    """
    raw = (title or "").strip()
    t0 = re.sub(r"\s*\(\d{4}\)\s*$", "", raw).strip()
    t1 = re.sub(r"\s*\(a\.k\.a\.\s*[^)]+\)\s*", " ", t0, flags=re.IGNORECASE)
    t1 = re.sub(r"\s+", " ", t1).strip()

    seen = set()

    def add(s):
        s = s.strip() if s else ""
        if len(s) < 2:
            return
        if s not in seen:
            seen.add(s)
            return s
        return None

    candidates = [t0, t1]
    m_comma = re.match(r"^(.+),\s*The$", t1, flags=re.IGNORECASE)
    if m_comma:
        candidates.append(f"The {m_comma.group(1).strip()}")

    # "Dr. Strangelove or: How I Learned ..." -> "Dr. Strangelove"
    m_or = re.split(r"\s+or:\s+", t1, maxsplit=1, flags=re.IGNORECASE)
    if len(m_or) > 1 and m_or[0].strip():
        candidates.append(m_or[0].strip())
    m_or2 = re.split(r"\s+ or \s+", t1, maxsplit=1, flags=re.IGNORECASE)
    if len(m_or2) > 1 and m_or2[0].strip() and m_or2[0] not in candidates:
        candidates.append(m_or2[0].strip())

    for t in candidates:
        s = add(t)
        if s is not None:
            yield s


def _result_release_year(item):
    rd = item.get("release_date") or ""
    if len(rd) >= 4 and rd[:4].isdigit():
        return int(rd[:4])
    return None


def _score_result(hit, target_year):
    """
    Higher is better. Prefer poster, year match to ML-1M (or ±1 for catalogue drift),
    then popularity.
    """
    score = 0.0
    if hit.get("poster_path"):
        score += 2000.0
    y = _result_release_year(hit)
    if target_year and y is not None:
        try:
            ty = int(target_year)
        except (TypeError, ValueError):
            ty = None
        if ty is not None:
            if y == ty:
                score += 500.0
            elif abs(y - ty) == 1:
                score += 200.0
    try:
        score += float(hit.get("popularity", 0) or 0)
    except (TypeError, ValueError):
        pass
    try:
        score += 0.01 * float(hit.get("vote_count", 0) or 0)
    except (TypeError, ValueError):
        pass
    return score


def _pick_best_hit(results, target_year, require_poster=True):
    """Pick one TMDB `results[]` item; prefer rows with a poster when require_poster."""
    if not results:
        return None
    pool = [h for h in results if h.get("poster_path")]
    if not pool and not require_poster:
        pool = list(results)
    if not pool:
        return None
    return max(pool, key=lambda h: _score_result(h, target_year))


def _search_request(clean_title, year):
    """
    One GET /search/movie. `year` None = do not send year (broader; needed when
    ML-1M year disagrees with TMDB primary year, e.g. Dr. Strangelove 1963 vs 1964).
    """
    params = {
        "api_key": Config.TMDB_API_KEY,
        "query": clean_title,
        "language": "en-US",
        "page": 1,
    }
    if year is not None:
        params["year"] = int(year)
    for attempt in range(2):
        try:
            resp = _tmdb_get(
                f"{Config.TMDB_BASE_URL}/search/movie",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError, KeyError):
            if attempt == 0:
                time.sleep(_TMDB_BETWEEN_ATTEMPTS_S)
    return None


def _search_tmdb(title, year):
    """Query TMDB /search/movie; try title variants, with- and without-year."""
    if not Config.TMDB_API_KEY:
        return None

    for clean_title in _ml_title_to_search_queries(title):
        time.sleep(_TMDB_BETWEEN_ATTEMPTS_S)
        if year is not None:
            data = _search_request(clean_title, year)
            results = (data or {}).get("results") or []
            hit = _pick_best_hit(results, year, require_poster=True)
            if hit:
                return hit
            time.sleep(_TMDB_BETWEEN_ATTEMPTS_S)
            # ML-1M year often ≠ TMDB primary year; with-year can return [] or the wrong row
            data = _search_request(clean_title, None)
            results = (data or {}).get("results") or []
            hit = _pick_best_hit(results, year, require_poster=True)
            if hit:
                return hit
        else:
            data = _search_request(clean_title, None)
            results = (data or {}).get("results") or []
            hit = _pick_best_hit(results, None, require_poster=True)
            if hit:
                return hit
            if results:
                hit = _pick_best_hit(results, None, require_poster=False)
                if hit:
                    return hit

    return None


def get_movie_poster_and_overview(movie_id, title, year):
    """
    Return (poster_url, overview) for a movie.
    Checks DB cache first; falls back to TMDB API if empty.
    Updates DB cache after fetch.
    """
    conn = get_db()
    row = conn.execute(
        "SELECT tmdb_poster_url, tmdb_overview FROM movies WHERE movie_id = ?",
        (movie_id,),
    ).fetchone()

    if row and row["tmdb_poster_url"]:
        conn.close()
        return row["tmdb_poster_url"], row["tmdb_overview"]

    # Fetch from TMDB
    result = _search_tmdb(title, year)
    poster_url = ""
    overview = ""

    if result:
        poster_path = result.get("poster_path") or ""
        if poster_path:
            poster_url = f"{Config.TMDB_IMAGE_BASE}{poster_path}"
        overview = result.get("overview", "")

    # Cache in DB (even if empty — avoids repeat misses)
    conn.execute(
        "UPDATE movies SET tmdb_poster_url = ?, tmdb_overview = ? WHERE movie_id = ?",
        (poster_url, overview, movie_id),
    )
    conn.commit()
    conn.close()

    return poster_url, overview


def enrich_movie_dict(movie):
    """Add/refresh poster_url and overview in a movie dict."""
    poster_url, overview = get_movie_poster_and_overview(
        movie["movie_id"], movie["title"], movie.get("year")
    )
    movie["tmdb_poster_url"] = poster_url
    movie["tmdb_overview"] = overview
    if not poster_url:
        movie["tmdb_poster_url"] = "/static/img/no_poster.png"
    return movie


def search_tmdb_suggestions(query: str, limit: int = 10):
    """
    Live TMDb /search/movie for autocomplete. Does not write to the DB.
    Use with find_catalog_match_for_external_title() to get ml-1m movie_id.
    """
    q = (query or "").strip()
    if not Config.TMDB_API_KEY or len(q) < 2:
        return []
    try:
        resp = _tmdb_get(
            f"{Config.TMDB_BASE_URL}/search/movie",
            params={
                "api_key": Config.TMDB_API_KEY,
                "query": q,
                "language": "en-US",
                "include_adult": "false",
                "page": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return []
    out = []
    for item in (data.get("results") or [])[:limit]:
        tid = item.get("id")
        title = (item.get("title") or item.get("original_title") or "").strip()
        if not title or not tid:
            continue
        rd = item.get("release_date") or ""
        year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
        p = item.get("poster_path")
        poster_url = f"{Config.TMDB_IMAGE_BASE}{p}" if p else None
        try:
            vot = float(item.get("vote_average") or 0)
        except (TypeError, ValueError):
            vot = 0.0
        out.append(
            {
                "tmdb_id": int(tid),
                "title": title,
                "year": year,
                "poster_url": poster_url,
                "vote_average": round(vot, 1),
            }
        )
    return out


def discover_tmdb_movies_modern(min_year: int = None, page: int = 1):
    """
    TMDb /discover/movie with primary release dates from min_year-01-01 through this year.
    Popularity-sorted, for hybrid browse alongside the ml-1m catalog.

    Returns: (list of {tmdb_id, title, year, poster_url}, total_pages, err_or_none)
    """
    if not (Config.TMDB_API_KEY or "").strip():
        return [], 0, "TMDB_API_KEY is not set."
    if min_year is None:
        min_year = getattr(Config, "TMDB_MODERN_MIN_YEAR", 2000)
    try:
        min_year = int(min_year)
    except (TypeError, ValueError):
        min_year = 2000
    min_year = max(1900, min(min_year, 2100))
    try:
        pg = int(page)
    except (TypeError, ValueError):
        pg = 1
    pg = max(1, min(pg, 500))

    end_y = date.today().year
    try:
        r = _tmdb_get(
            f"{Config.TMDB_BASE_URL}/discover/movie",
            params={
                "api_key": Config.TMDB_API_KEY,
                "language": "en-US",
                "page": pg,
                "primary_release_date.gte": f"{min_year}-01-01",
                "primary_release_date.lte": f"{end_y}-12-31",
                "sort_by": "popularity.desc",
                "include_adult": "false",
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json() or {}
    except (requests.RequestException, TypeError, ValueError):
        return [], 0, "TMDb discover request failed. Try again or check the API key."

    out = []
    for item in data.get("results") or []:
        tid = item.get("id")
        title = (item.get("title") or item.get("original_title") or "").strip()
        if not title or not tid:
            continue
        rd = item.get("release_date") or ""
        year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
        p = item.get("poster_path")
        poster_url = f"{Config.TMDB_IMAGE_BASE}{p}" if p else None
        out.append(
            {
                "tmdb_id": int(tid),
                "title": title,
                "year": year,
                "poster_url": poster_url or "",
            }
        )
    tp = int(data.get("total_pages") or 1)
    return out, max(1, tp), None


# TMDb genre names -> MovieLens-1M pipe tokens (kept in sync with recommender ALL_GENRES)
_ML1M_GENRE_SET = frozenset(
    {
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    }
)
_TMD_NAME_TO_ML = {
    "Science Fiction": "Sci-Fi",
    "Sci-Fi & Fantasy": "Sci-Fi",
    "Action & Adventure": "Action",  # pair both below
}


def _tmdb_to_ml_genre_pipe(tmdb_genre_objects):
    """Map TMDb /movie/{id} genres[] to a pipe string matching ml-1m."""
    if not tmdb_genre_objects:
        return "Drama"
    ml = set()
    for g in tmdb_genre_objects:
        name = (g.get("name") if isinstance(g, dict) else str(g) or "").strip()
        if not name:
            continue
        n = _TMD_NAME_TO_ML.get(name, name)
        if n in _ML1M_GENRE_SET:
            ml.add(n)
            continue
        if n in ("Action & Adventure",) or (name and "Action" in name and "Adventure" in name):
            ml.add("Action")
            ml.add("Adventure")
        elif n in ("TV Movie", "History", "Western", "War"):
            ml.add("Drama")
    if not ml:
        ml.add("Drama")
    return "|".join(sorted(ml))


def fetch_tmdb_movie_for_import(tmdb_id):
    """
    Full TMDb movie details for persisting a supplement row. No DB write.
    Returns (data_dict, None) on success, or (None, error_message) on failure.
    """
    if not (Config.TMDB_API_KEY or "").strip():
        return None, (
            "TMDB_API_KEY is missing. Add it to the project .env file and restart "
            "Gunicorn (or Flask) so the server reloads the environment."
        )
    try:
        tid = int(tmdb_id)
    except (TypeError, ValueError):
        return None, "Invalid tmdb_id."
    if tid < 1:
        return None, "Invalid tmdb_id."
    try:
        r = _tmdb_get(
            f"{Config.TMDB_BASE_URL}/movie/{tid}",
            params={"api_key": Config.TMDB_API_KEY, "language": "en-US"},
            timeout=12,
        )
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    ) as e:
        return None, (
            "Could not reach TMDb after several tries ("
            f"{type(e).__name__}). Check your network, VPN, or firewall, then try again. "
            "Connection resets to api.themoviedb.org are often transient."
        )[:500]
    except requests.RequestException as e:
        return None, f"TMDb request failed: {e!s}"[:300]

    if r.status_code == 401:
        return None, (
            "TMDb rejected the API key (401). Check TMDB_API_KEY in .env and restart the server. "
            "Get a v3 key at themoviedb.org/settings/api."
        )
    if r.status_code == 404:
        return None, "This movie was not found on TMDb (404)."
    if not r.ok:
        try:
            j = r.json()
            err = j.get("status_message", j.get("errors", r.text))[:200]
        except (ValueError, TypeError, AttributeError):
            err = (r.text or "")[:200]
        return None, f"TMDb error {r.status_code}: {err}"

    try:
        d = r.json()
    except ValueError:
        return None, "Invalid response from TMDb."

    title = (d.get("title") or d.get("original_title") or "").strip()
    if not title:
        return None, "TMDb returned no title for this id."
    rd = d.get("release_date") or ""
    year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
    p = d.get("poster_path")
    poster_url = f"{Config.TMDB_IMAGE_BASE}{p}" if p else ""
    gpipe = _tmdb_to_ml_genre_pipe(d.get("genres") or [])
    overview = (d.get("overview") or "").strip()
    return {
        "tmdb_id": int(tid),
        "title": title,
        "year": year,
        "genres_pipe": gpipe,
        "poster_url": poster_url,
        "overview": overview,
    }, None


def resolve_tmdb_id_for_cinematch_row(row) -> int | None:
    """
    TMDb numeric id for a DB movie row: use stored tmdb_id or search by title/year.
    """
    if not row:
        return None
    if row.get("tmdb_id"):
        try:
            return int(row["tmdb_id"])
        except (TypeError, ValueError):
            pass
    hit = _search_tmdb(row.get("title", ""), row.get("year"))
    if hit and hit.get("id"):
        return int(hit["id"])
    return None


def fetch_tmdb_similar_and_recommendations(
    tmdb_id: int, exclude_tmdb_id: int | None = None, per_endpoint: int = 10
):
    """
    Merge /movie/{id}/similar and /movie/{id}/recommendations. Returns (list, error_or_none).
    """
    if not (Config.TMDB_API_KEY or "").strip():
        return [], "TMDB_API_KEY is not set."
    tid = int(tmdb_id)
    out: list[dict] = []
    seen: set[int] = {tid}
    if exclude_tmdb_id is not None:
        seen.add(int(exclude_tmdb_id))
    for path in ("similar", "recommendations"):
        time.sleep(_TMDB_BETWEEN_ATTEMPTS_S)
        try:
            r = _tmdb_get(
                f"{Config.TMDB_BASE_URL}/movie/{tid}/{path}",
                params={
                    "api_key": Config.TMDB_API_KEY,
                    "language": "en-US",
                    "page": 1,
                },
                timeout=12,
            )
            r.raise_for_status()
            data = r.json() or {}
        except (requests.RequestException, TypeError, ValueError):
            continue
        for item in (data.get("results") or [])[:per_endpoint]:
            iid = item.get("id")
            if not iid or int(iid) in seen:
                continue
            seen.add(int(iid))
            title = (
                item.get("title")
                or item.get("name")
                or item.get("original_title")
                or ""
            ).strip()
            if not title:
                continue
            rd = item.get("release_date") or ""
            year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
            p = item.get("poster_path")
            poster_url = f"{Config.TMDB_IMAGE_BASE}{p}" if p else None
            out.append(
                {
                    "tmdb_id": int(iid),
                    "title": title,
                    "year": year,
                    "poster_url": poster_url or "",
                }
            )
    if not out:
        return (
            [],
            "No related titles from TMDb (or request failed). Check the API key and try again.",
        )
    return out, None
