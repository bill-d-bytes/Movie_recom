"""
app.py — CineMatch Flask Application
All routes: auth, pages, API endpoints.
"""
import json
import math
import os
import time
import functools
from flask import (Flask, render_template, request, session,
                   redirect, url_for, jsonify, flash, g, current_app, send_from_directory)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_caching import Cache

from config import Config, ml_models_are_built
from database import (get_db, create_tables, seed_movies, seed_ratings, ensure_movies_migrated,
                      load_movies_df, load_ratings_df, find_catalog_match_for_external_title,
                      get_movie_by_id, search_movies, get_trending_movies, get_avg_ratings_for_movies,
                      get_latest_movies, get_movie_by_tmdb_id, insert_tmdb_supplement)
from recommender import RecommendationEngine, ERA_YEAR_RANGES
from tmdb import (
    discover_tmdb_movies_modern,
    discover_tmdb_movies_by_language,
    enrich_movie_dict,
    fetch_tmdb_movie_for_import,
    fetch_tmdb_similar_and_recommendations,
    resolve_tmdb_id_for_cinematch_row,
    search_tmdb_suggestions,
)

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app   = Flask(__name__, template_folder='app/templates',
              static_folder='app/static')
app.config.from_object(Config)

cache = Cache(app)

# Singleton recommendation engine (loaded once)
engine = RecommendationEngine()


# ──────────────────────────────────────────────
# App startup — init DB + load models
# ──────────────────────────────────────────────
def init_app():
    conn = get_db()
    create_tables(conn)
    conn.close()
    ensure_movies_migrated()
    conn = get_db()
    # Seed dataset if DB is empty
    movies_df  = load_movies_df()
    ratings_df = load_ratings_df()
    seed_movies(conn, movies_df)
    seed_ratings(conn, ratings_df)
    conn.close()

    if ml_models_are_built():
        engine.load()
    else:
        print("⚠️  ML models not found. Run: python recommender.py --build")


with app.app_context():
    init_app()


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        app.static_folder,
        "favicon.svg",
        mimetype="image/svg+xml",
    )


# ──────────────────────────────────────────────
# API timing (optional: set CINEMATCH_LOG_SLOW_API_MS=2000 to log slow /api/*)
# ──────────────────────────────────────────────
@app.before_request
def _api_timer_start():
    if request.path.startswith('/api/'):
        g._api_t0 = time.perf_counter()


@app.after_request
def _log_slow_api(response):
    if not request.path.startswith('/api/'):
        return response
    threshold = os.environ.get('CINEMATCH_LOG_SLOW_API_MS')
    if not threshold:
        return response
    try:
        ms_limit = int(threshold)
    except ValueError:
        return response
    t0 = getattr(g, '_api_t0', None)
    if t0 is not None:
        ms = (time.perf_counter() - t0) * 1000.0
        if ms >= ms_limit:
            current_app.logger.warning(
                'Slow API %s %s → %d bytes in %.0fms',
                request.method, request.path, response.content_length or 0, ms
            )
    return response


# ──────────────────────────────────────────────
# Auth decorator
# ──────────────────────────────────────────────
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Unauthorised'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def _safe_year_value(y):
    """SQLite / pandas year for JSON; avoid int(nan) and similar."""
    if y is None:
        return None
    try:
        f = float(y)
        if math.isnan(f) or math.isinf(f):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return int(f)
    except (TypeError, ValueError, OverflowError):
        return None


def _sanitize_for_json(obj):
    """
    Make nested structures safe for jsonify: replace NaN/Inf with None,
    convert numpy / pandas scalars to native Python.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return _sanitize_for_json(obj.item())
        except (TypeError, ValueError, AttributeError, OverflowError):
            return None
    return str(obj) if obj is not None else None


# ML-1M / MovieLens genre labels → TMDb genre IDs
# Used when calling TMDb discover so results match the user's genre preferences.
_ML_GENRE_TO_TMDB_ID = {
    "action":      28,
    "adventure":   12,
    "animation":   16,
    "children's":  10751,
    "children":    10751,
    "comedy":      35,
    "crime":       80,
    "documentary": 99,
    "drama":       18,
    "fantasy":     14,
    "film-noir":   53,   # closest TMDb equivalent: Thriller
    "horror":      27,
    "musical":     10402,
    "mystery":     9648,
    "romance":     10749,
    "sci-fi":      878,
    "science fiction": 878,
    "thriller":    53,
    "war":         10752,
    "western":     37,
}


def _genres_to_tmdb_ids(genres: list) -> list:
    """Map a list of ML-style genre strings to TMDb genre IDs (deduped)."""
    seen, out = set(), []
    for g in (genres or []):
        tid = _ML_GENRE_TO_TMDB_ID.get((g or "").strip().lower())
        if tid and tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def _merge_latest_into_recommendations(
    hybrid_rows,
    top_n,
    anchor_ids,
    era_filter,
    min_avg_f,
    exclude_tmdb_supplements_from_latest: bool = False,
):
    """
    Mix in a few of the newest (by release year) catalog titles, respecting the same
    era and min rating constraints as the hybrid list. Puts rec_source: 'latest' on those.
    If exclude_tmdb_supplements_from_latest, only ml1m rows qualify (no recent TMDb-only imports).
    """
    L = getattr(Config, "LATEST_IN_RECOMMENDATION", 0) or 0
    if L <= 0 or top_n <= 0:
        return hybrid_rows[:top_n] if hybrid_rows else [], 0

    hybrid_sorted = sorted(
        hybrid_rows or [],
        key=lambda x: x.get("hybrid_score", 0),
        reverse=True,
    )
    anchor_ids = {int(x) for x in (anchor_ids or [])}
    latest_quota = min(L, top_n)
    hybrid_quota = max(0, top_n - latest_quota)
    out = []
    seen = set(anchor_ids)

    for r in hybrid_sorted:
        if len(out) >= hybrid_quota:
            break
        mid = int(r["movie_id"])
        if mid in seen:
            continue
        seen.add(mid)
        out.append(r)

    pool = get_latest_movies(
        limit=max(50, top_n * 8),
        min_year=Config.LATEST_MIN_YEAR,
        exclude_tmdb_supplements=exclude_tmdb_supplements_from_latest,
    )
    avgs = {}
    if min_avg_f is not None and min_avg_f > 0 and pool:
        avgs = get_avg_ratings_for_movies([int(p["movie_id"]) for p in pool])

    ef = era_filter or []

    def _year_in_filter(y):
        if y is None:
            return False
        if not ef:
            return True
        try:
            yi = int(y)
        except (TypeError, ValueError):
            return False
        return any(
            ERA_YEAR_RANGES[e][0] <= yi <= ERA_YEAR_RANGES[e][1]
            for e in ef
            if e in ERA_YEAR_RANGES
        )

    n_added = 0
    for row in pool:
        if len(out) >= top_n:
            break
        if n_added >= latest_quota:
            break
        mid = int(row["movie_id"])
        if mid in seen:
            continue
        y = row.get("year")
        if not _year_in_filter(y):
            continue
        if min_avg_f is not None and min_avg_f > 0:
            ar = avgs.get(mid)
            if ar is not None and ar < min_avg_f:
                continue
        seen.add(mid)
        g = row.get("genres", "") or "Drama"
        genres = g.split("|") if isinstance(g, str) else (g or ["Drama"])
        y_out = _safe_year_value(y)
        out.append(
            {
                "movie_id": mid,
                "title": row["title"],
                "genres": [x.strip() for x in genres if x.strip()] or ["Drama"],
                "year": y_out,
                "hybrid_score": 0.0,
                "content_score": 0.0,
                "collab_score": 0.0,
                "persona_score": 0.0,
                "is_cold_start": False,
                "rec_source": "latest",
            }
        )
        n_added += 1

    for r in hybrid_sorted:
        if len(out) >= top_n:
            break
        mid = int(r["movie_id"])
        if mid in seen:
            continue
        seen.add(mid)
        out.append(r)

    return out[:top_n], n_added


def _merge_regional_into_recommendations(
    rows,
    top_n,
    anchor_ids,
    era_filter,
    min_avg_f,
    user_profile=None,
    anchor_genres=None,
):
    """
    Replace up to REGIONAL_IN_RECOMMENDATION trailing slots with popular TMDb discovers
    in original languages hi / te / ta (Bollywood, Telugu, Tamil), imported into the catalog.
    When user_profile / anchor_genres are supplied, genre-filters the TMDb discover call.
    """
    R = getattr(Config, "REGIONAL_IN_RECOMMENDATION", 0) or 0
    if R <= 0 or top_n <= 0 or not (Config.TMDB_API_KEY or "").strip():
        return (rows or [])[:top_n], 0

    rows = list(rows or [])

    regional_quota = min(R, top_n)
    keep_n = max(0, top_n - regional_quota)
    base = rows[:keep_n]
    anchor_ids = {int(x) for x in (anchor_ids or [])}
    seen = {int(r["movie_id"]) for r in base} | anchor_ids

    ef = era_filter or []

    def _year_in_filter(y):
        if y is None:
            return False
        if not ef:
            return True
        try:
            yi = int(y)
        except (TypeError, ValueError):
            return False
        return any(
            ERA_YEAR_RANGES[e][0] <= yi <= ERA_YEAR_RANGES[e][1]
            for e in ef
            if e in ERA_YEAR_RANGES
        )

    min_regional_year = getattr(Config, "REGIONAL_MIN_YEAR", 2000)
    langs = getattr(Config, "REGIONAL_ORIGINAL_LANGUAGES", ("hi", "te", "ta"))

    # Build TMDb genre filter from anchor + user preferences
    genre_pool = list(anchor_genres or [])
    if user_profile:
        genre_pool += list(user_profile.get("preferred_genres") or [])
    regional_genre_ids = _genres_to_tmdb_ids(genre_pool) or None

    per_lang = []
    for lang in langs:
        items, _, err = discover_tmdb_movies_by_language(
            lang, min_year=min_regional_year, page=1,
            genre_ids=regional_genre_ids,
        )
        if err or not items:
            # Fallback: no genre filter if genre-filtered call found nothing
            items, _, err = discover_tmdb_movies_by_language(
                lang, min_year=min_regional_year, page=1
            )
        if err or not items:
            continue
        per_lang.append([(lang, it) for it in items[:12]])

    candidates = []
    idx = 0
    while True:
        progressed = False
        for lst in per_lang:
            if idx < len(lst):
                candidates.append(lst[idx])
                progressed = True
        if not progressed:
            break
        idx += 1

    regional_rows = []
    for lang, it in candidates:
        if len(regional_rows) >= regional_quota:
            break
        tid = int(it["tmdb_id"])
        row_db = get_movie_by_tmdb_id(tid)
        if row_db:
            mid = int(row_db["movie_id"])
            y = row_db.get("year")
            if not _year_in_filter(y):
                continue
            title = row_db["title"]
            g = row_db.get("genres", "") or "Drama"
        else:
            full, err = fetch_tmdb_movie_for_import(tid)
            if err or not full:
                continue
            y = full.get("year")
            if not _year_in_filter(y):
                continue
            mid = int(
                insert_tmdb_supplement(
                    full["tmdb_id"],
                    full["title"],
                    full["year"],
                    full["genres_pipe"],
                    full["poster_url"],
                    full["overview"],
                )
            )
            title = full["title"]
            g = full.get("genres_pipe") or "Drama"

        if mid in seen:
            continue
        if min_avg_f is not None and min_avg_f > 0:
            avgs = get_avg_ratings_for_movies([mid])
            ar = avgs.get(mid)
            if ar is not None and ar < min_avg_f:
                continue

        seen.add(mid)
        genres = (
            [x.strip() for x in g.split("|") if x.strip()]
            if isinstance(g, str)
            else (g or ["Drama"])
        ) or ["Drama"]
        y_out = _safe_year_value(y)
        regional_rows.append(
            {
                "movie_id": mid,
                "title": title,
                "genres": genres,
                "year": y_out,
                "hybrid_score": 0.0,
                "content_score": 0.0,
                "collab_score": 0.0,
                "persona_score": 0.0,
                "is_cold_start": False,
                "rec_source": "regional_in",
                "regional_lang": lang,
            }
        )

    out = list(base) + regional_rows
    seen = {int(r["movie_id"]) for r in out} | anchor_ids
    for r in rows:
        if len(out) >= top_n:
            break
        mid = int(r["movie_id"])
        if mid in seen:
            continue
        seen.add(mid)
        out.append(r)

    return out[:top_n], len(regional_rows)


def _merge_tmdb_popular_into_recommendations(
    rows,
    top_n,
    anchor_ids,
    era_filter,
    min_avg_f,
    user_profile=None,
    anchor_genres=None,
):
    """
    On ML-1M only: fill remaining slots with popular TMDb movies matching the era_filter
    AND the user's genre preferences (+ anchor movie genres).
    Triggered when the hybrid engine returned sparse/no results because ML-1M (max ~2000)
    cannot serve modern/00s-era queries.  Skipped entirely when running ML-25M.
    """
    if Config.DATASET_USE_25M:
        return (rows or [])[:top_n], 0
    if not (Config.TMDB_API_KEY or "").strip():
        return (rows or [])[:top_n], 0

    ef = era_filter or []
    if not ef:
        return (rows or [])[:top_n], 0

    # Compute the year window from the requested era chips
    valid_eras = [e for e in ef if e in ERA_YEAR_RANGES]
    if not valid_eras:
        return (rows or [])[:top_n], 0
    era_min = min(ERA_YEAR_RANGES[e][0] for e in valid_eras)
    era_max = max(ERA_YEAR_RANGES[e][1] for e in valid_eras)

    # Only fill for eras that are beyond what ML-1M carries (~2000)
    if era_min < 2001:
        return (rows or [])[:top_n], 0

    rows = list(rows or [])
    needed = top_n - len(rows)
    if needed <= 0:
        return rows[:top_n], 0

    # Build TMDb genre filter from user preferred genres + anchor movie genres
    genre_pool = list(anchor_genres or [])
    if user_profile:
        genre_pool += list(user_profile.get("preferred_genres") or [])
    tmdb_genre_ids = _genres_to_tmdb_ids(genre_pool) or None

    # Map min_avg_f (1–5 scale) to TMDb vote_average (0–10 scale)
    vote_gte = None
    if min_avg_f is not None:
        try:
            vote_gte = float(min_avg_f) * 2.0
        except (TypeError, ValueError):
            pass

    items, _, err = discover_tmdb_movies_modern(
        min_year=era_min, max_year=era_max, page=1,
        genre_ids=tmdb_genre_ids, vote_average_gte=vote_gte,
    )
    if err or not items:
        # Fallback: retry without genre filter (user may have niche combo)
        items, _, err = discover_tmdb_movies_modern(
            min_year=era_min, max_year=era_max, page=1,
        )
    if err or not items:
        return rows[:top_n], 0

    anchor_ids = {int(x) for x in (anchor_ids or [])}
    seen = {int(r["movie_id"]) for r in rows} | anchor_ids
    out = list(rows)
    n_added = 0

    for it in items:
        if len(out) >= top_n:
            break
        tid = int(it["tmdb_id"])
        row_db = get_movie_by_tmdb_id(tid)
        if row_db:
            mid = int(row_db["movie_id"])
            if mid in seen:
                continue
            y = row_db.get("year")
            title = row_db["title"]
            g = row_db.get("genres", "") or "Drama"
        else:
            full, err2 = fetch_tmdb_movie_for_import(tid)
            if err2 or not full:
                continue
            y = full.get("year")
            mid = int(
                insert_tmdb_supplement(
                    full["tmdb_id"],
                    full["title"],
                    full["year"],
                    full["genres_pipe"],
                    full["poster_url"],
                    full["overview"],
                )
            )
            if mid in seen:
                continue
            title = full["title"]
            g = full.get("genres_pipe") or "Drama"

        seen.add(mid)
        genres = (
            [x.strip() for x in g.split("|") if x.strip()]
            if isinstance(g, str)
            else (g or ["Drama"])
        ) or ["Drama"]
        y_out = _safe_year_value(y)
        out.append(
            {
                "movie_id": mid,
                "title": title,
                "genres": genres,
                "year": y_out,
                "hybrid_score": 0.0,
                "content_score": 0.0,
                "collab_score": 0.0,
                "persona_score": 0.0,
                "is_cold_start": False,
                "rec_source": "tmdb_popular",
            }
        )
        n_added += 1

    return out[:top_n], n_added


# ──────────────────────────────────────────────
# PAGE ROUTES  (serve Jinja2 templates)
# ──────────────────────────────────────────────
@app.route('/')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('discover_page'))
    return render_template('login.html')


@app.route('/discover')
@login_required
def discover_page():
    return render_template('discover.html')


@app.route('/profile')
@login_required
def profile_page():
    return render_template('profile.html')


@app.route('/preferences')
@login_required
def preferences_page():
    return redirect('/discover#discover-preferences')


@app.route('/filters')
@login_required
def filters_page():
    return redirect('/discover#discover-precision')


@app.route('/movie/<int:movie_id>')
@login_required
def movie_detail_page(movie_id):
    return render_template('movie_detail.html', movie_id=movie_id)


# ──────────────────────────────────────────────
# AUTH ROUTES
# ──────────────────────────────────────────────
@app.route('/auth/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('login.html', mode='register')

    data = request.get_json() or request.form.to_dict()
    username = (data.get('username') or '').strip()
    email    = (data.get('email') or '').strip()
    password = data.get('password') or ''

    if not username or not email or not password:
        return jsonify({'error': 'All fields required'}), 400

    conn = get_db()
    existing = conn.execute(
        "SELECT user_id FROM app_users WHERE username = ? OR email = ?",
        (username, email)
    ).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': 'Username or email already taken'}), 409

    # pbkdf2: sha256 is portable; scrypt (Werkzeug default) requires OpenSSL with scrypt support
    pw_hash = generate_password_hash(password, method="pbkdf2:sha256")
    cursor = conn.execute(
        "INSERT INTO app_users (username, email, password_hash) VALUES (?, ?, ?)",
        (username, email, pw_hash)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    session['user_id']  = user_id
    session['username'] = username
    return jsonify({'success': True, 'redirect': '/discover'}), 201


@app.route('/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json() or request.form.to_dict()
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM app_users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid username or password'}), 401

    session['user_id']  = user['user_id']
    session['username'] = user['username']
    return jsonify({'success': True, 'redirect': '/discover'})


@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    session.clear()
    return jsonify({'success': True, 'redirect': '/'})


# ──────────────────────────────────────────────
# API — PROFILE
# ──────────────────────────────────────────────
@app.route('/api/profile', methods=['GET', 'PUT'])
@login_required
def api_profile():
    user_id = session['user_id']
    conn = get_db()

    if request.method == 'GET':
        row = conn.execute(
            "SELECT user_id, username, email, age, gender, location, dob, "
            "preferred_genres, notifications FROM app_users WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'User not found'}), 404
        data = dict(row)
        data['preferred_genres'] = json.loads(data['preferred_genres'] or '[]')
        return jsonify(data)

    # PUT — update profile fields
    data = request.get_json() or {}
    allowed = ['email', 'age', 'gender', 'location', 'dob', 'notifications']
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        conn.close()
        return jsonify({'error': 'No valid fields to update'}), 400

    set_clause = ', '.join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [user_id]
    conn.execute(f"UPDATE app_users SET {set_clause} WHERE user_id = ?", values)
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ──────────────────────────────────────────────
# API — PREFERENCES
# ──────────────────────────────────────────────
@app.route('/api/preferences', methods=['GET', 'PUT'])
@login_required
def api_preferences():
    user_id = session['user_id']
    conn = get_db()

    if request.method == 'GET':
        row = conn.execute(
            "SELECT preferred_genres, age, gender FROM app_users WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'User not found'}), 404
        return jsonify({
            'preferred_genres': json.loads(row['preferred_genres'] or '[]'),
            'age':    row['age'],
            'gender': row['gender'],
        })

    data = request.get_json() or {}
    genres = data.get('preferred_genres', [])
    age    = data.get('age')
    gender = data.get('gender')

    conn.execute(
        "UPDATE app_users SET preferred_genres = ?, age = ?, gender = ? WHERE user_id = ?",
        (json.dumps(genres), age, gender, user_id)
    )
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ──────────────────────────────────────────────
# API — MOVIES
# ──────────────────────────────────────────────
@app.route('/api/movies/search')
@login_required
def api_movie_search():
    q = (request.args.get('q') or '').strip()
    if len(q) < 2:
        return jsonify([])
    results = search_movies(q, limit=20)
    return jsonify(results)


@app.route('/api/movies/suggest')
@login_required
def api_movie_suggest():
    """
    Autocomplete: local ml-1m matches + TMDb search with fuzzy link to catalog when possible.
    Titles not in the library appear with in_catalog false (recommendations still need an ml-1m id).
    """
    q = (request.args.get('q') or '').strip()
    if len(q) < 2:
        return jsonify({'results': []})
    out = []
    seen = set()
    for r in search_movies(q, limit=20):
        mid = r['movie_id']
        if mid in seen:
            continue
        seen.add(mid)
        out.append({
            'source': 'local',
            'movie_id': mid,
            'title': r['title'],
            'year': r['year'],
            'genres': r.get('genres', ''),
            'in_catalog': True,
        })
    for t in search_tmdb_suggestions(q, limit=10):
        mid = find_catalog_match_for_external_title(t['title'], t.get('year'))
        if not mid:
            row_t = get_movie_by_tmdb_id(int(t['tmdb_id']))
            if row_t:
                mid = row_t['movie_id']
        if mid and mid in seen:
            continue
        item = {
            'source': 'tmdb',
            'title': t['title'],
            'year': t.get('year'),
            'tmdb_id': t['tmdb_id'],
            'poster_url': t.get('poster_url'),
            'vote_average': t.get('vote_average', 0),
        }
        if mid:
            item['movie_id'] = mid
            item['in_catalog'] = True
            seen.add(mid)
        else:
            item['movie_id'] = None
            item['in_catalog'] = False
        out.append(item)
    return jsonify({'results': out})


@app.route('/api/movies/from-tmdb', methods=['POST'])
@login_required
def api_movie_from_tmdb():
    """Import a TMDb-only title into the local catalog (synthetic movie_id). Idempotent on tmdb_id."""
    d = request.get_json(silent=True) or {}
    raw = d.get("tmdb_id")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return jsonify({"error": "tmdb_id is required in the JSON body."}), 400
    try:
        tmdb_id = int(raw)
    except (TypeError, ValueError):
        return jsonify({"error": "tmdb_id must be a whole number (TMDb movie id)."}), 400
    if tmdb_id < 1:
        return jsonify({"error": "tmdb_id must be a positive TMDb movie id."}), 400
    ex = get_movie_by_tmdb_id(tmdb_id)
    if ex:
        m = dict(ex)
    else:
        info, tmdb_err = fetch_tmdb_movie_for_import(tmdb_id)
        if tmdb_err or not info:
            return jsonify({"error": tmdb_err or "Could not load that title from TMDb."}), 400
        mid = insert_tmdb_supplement(
            tmdb_id=info['tmdb_id'],
            title=info['title'],
            year=info['year'],
            genres_pipe=info['genres_pipe'],
            poster_url=info.get('poster_url', ''),
            overview=info.get('overview', ''),
        )
        m = get_movie_by_id(mid)
    m = enrich_movie_dict(m) if m else None
    if not m:
        return jsonify({'error': 'Import failed'}), 500
    m['genres'] = m['genres'].split('|') if isinstance(m.get('genres'), str) else m.get('genres', [])
    return jsonify({
        'movie_id': m['movie_id'],
        'movie':    m,
    })


@app.route("/api/movies/modern-tmdb")
@login_required
def api_movies_modern_tmdb():
    """
    Hybrid: TMDb discover (2000+ by default) with optional link to a CineMatch row.
    Complements the ml-1m / SQLite trending grid on Discover.
    """
    page = request.args.get("page", 1, type=int) or 1
    page = max(1, min(page, 500))
    min_y = request.args.get("min_year", type=int)
    if min_y is None:
        min_y = Config.TMDB_MODERN_MIN_YEAR
    else:
        min_y = max(1900, min(int(min_y), 2100))

    cache_key = f"modern_tmdb_v1_{min_y}_p{page}"
    cached = cache.get(cache_key)
    if cached is not None:
        return jsonify(cached)

    items, total_pages, tmdb_err = discover_tmdb_movies_modern(min_year=min_y, page=page)
    if tmdb_err and not items:
        payload = {
            "source": "tmdb_discover",
            "min_year": min_y,
            "page": page,
            "total_pages": 0,
            "message": tmdb_err,
            "results": [],
        }
        cache.set(cache_key, payload, timeout=60)
        return jsonify(payload)

    results = []
    for it in items:
        tid = int(it["tmdb_id"])
        row_m = get_movie_by_tmdb_id(tid)
        mid = row_m["movie_id"] if row_m else find_catalog_match_for_external_title(
            it.get("title", ""), it.get("year")
        )
        results.append(
            {
                "tmdb_id": tid,
                "title": it.get("title", ""),
                "year": it.get("year"),
                "poster_url": it.get("poster_url") or "",
                "in_app_movie_id": int(mid) if mid is not None else None,
            }
        )
    payload = {
        "source": "tmdb_discover",
        "min_year": min_y,
        "page": page,
        "total_pages": int(total_pages) if total_pages else 1,
        "message": None,
        "results": results,
    }
    cache.set(cache_key, payload, timeout=Config.CACHE_DEFAULT_TIMEOUT)
    return jsonify(payload)


@app.route('/api/movies/trending')
@login_required
@cache.cached(timeout=600, key_prefix='trending')
def api_trending():
    movies = get_trending_movies(limit=20)
    # TMDB: enrich every row; cache in DB so repeat loads are fast
    for m in movies:
        enrich_movie_dict(m)
    return jsonify(movies)


@app.route('/api/movies/<int:movie_id>')
@login_required
def api_movie_detail(movie_id):
    movie = get_movie_by_id(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404
    movie = enrich_movie_dict(movie)
    movie['genres'] = movie['genres'].split('|') if isinstance(movie['genres'], str) else movie['genres']
    return jsonify(movie)


@app.route("/api/movies/tmdb-related")
@login_required
def api_tmdb_related():
    """
    TMDb similar + recommendations for the current title (broad, not MovieLens-hybrid).
    """
    movie_id = request.args.get("movie_id", type=int)
    if not movie_id or movie_id < 1:
        return jsonify({"error": "movie_id is required and must be a positive integer."}), 400
    row = get_movie_by_id(movie_id)
    if not row:
        return jsonify({"error": "Movie not found"}), 404

    tmdb_id = resolve_tmdb_id_for_cinematch_row(row)
    if not tmdb_id:
        return jsonify(
            {
                "source": "tmdb",
                "tmdb_id": None,
                "message": "Could not resolve a TMDb id for this title.",
                "results": [],
            }
        )

    cache_key = f"tmdb_related_v1_{movie_id}_{tmdb_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return jsonify(cached)

    items, tmdb_err = fetch_tmdb_similar_and_recommendations(
        tmdb_id, exclude_tmdb_id=tmdb_id, per_endpoint=10
    )
    if not items:
        payload = {
            "source": "tmdb",
            "tmdb_id": tmdb_id,
            "message": tmdb_err or "No related titles from TMDb.",
            "results": [],
        }
        cache.set(cache_key, payload, timeout=Config.CACHE_DEFAULT_TIMEOUT)
        return jsonify(payload)

    results = []
    for it in items:
        tid = int(it["tmdb_id"])
        row_m = get_movie_by_tmdb_id(tid)
        mid = row_m["movie_id"] if row_m else find_catalog_match_for_external_title(
            it.get("title", ""), it.get("year")
        )
        if mid is not None and int(mid) == int(movie_id):
            continue
        results.append(
            {
                "tmdb_id": tid,
                "title": it.get("title", ""),
                "year": it.get("year"),
                "poster_url": it.get("poster_url") or "",
                "in_app_movie_id": int(mid) if mid is not None else None,
            }
        )

    payload = {
        "source": "tmdb",
        "tmdb_id": tmdb_id,
        "message": None,
        "results": results,
    }
    cache.set(cache_key, payload, timeout=Config.CACHE_DEFAULT_TIMEOUT)
    return jsonify(payload)


# ──────────────────────────────────────────────
# API — RECOMMENDATIONS
# ──────────────────────────────────────────────
@app.route('/api/recommend', methods=['POST'])
@login_required
def api_recommend():
    user_id = session['user_id']
    data = request.get_json() or {}

    movie_id   = data.get('movie_id')
    era_filter = data.get('era_filter', [])   # e.g. ["90s", "00s"]
    top_n      = int(data.get('top_n', Config.TOP_N_DEFAULT))
    min_avg_rating = data.get('min_avg_rating', None)
    if min_avg_rating in (None, '', 'any'):
        min_avg_f = None
    else:
        try:
            min_avg_f = float(min_avg_rating)
        except (TypeError, ValueError):
            return jsonify({'error': 'min_avg_rating must be a number 0–5 (MovieLens scale)'}), 400
        if min_avg_f < 0 or min_avg_f > 5:
            return jsonify({'error': 'min_avg_rating must be between 0 and 5'}), 400
        if min_avg_f == 0:
            min_avg_f = None

    if not movie_id:
        return jsonify({'error': 'movie_id is required'}), 400

    include_latest = data.get("include_latest", True)
    if isinstance(include_latest, str):
        include_latest = str(include_latest).lower() in ("1", "true", "yes", "on")

    include_regional_indian = data.get("include_regional_indian", True)
    if isinstance(include_regional_indian, str):
        include_regional_indian = str(include_regional_indian).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    # Build user profile from DB
    conn = get_db()
    user_row = conn.execute(
        "SELECT age, gender, preferred_genres FROM app_users WHERE user_id = ?",
        (user_id,)
    ).fetchone()
    conn.close()

    if not user_row:
        return jsonify({'error': 'User not found'}), 404

    user_profile = {
        'age':              user_row['age'] or 25,
        'gender':           user_row['gender'] or 'M',
        'preferred_genres': json.loads(user_row['preferred_genres'] or '[]'),
    }

    if not engine._loaded:
        return jsonify({'error': 'Models not loaded yet. Run: python recommender.py --build'}), 503

    try:
        min_part = f"{min_avg_f!s}" if min_avg_f is not None else 'na'
        il_flag = 1 if include_latest else 0
        ir_flag = 1 if include_regional_indian else 0
        cache_key = (
            f"recommend_{user_id}_{movie_id}_{'-'.join(sorted(era_filter))}"
            f"_{min_part}_{top_n}_v9genreaware{il_flag}r{ir_flag}"
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return jsonify(_sanitize_for_json(cached))

        results, rec_meta = engine.hybrid_recommend(
            user_id=user_id,
            movie_id=movie_id,
            user_profile=user_profile,
            top_n=top_n,
            era_filter=era_filter or None,
            min_avg_rating=min_avg_f,
        )

        # Fetch anchor movie genres for genre-aware blends
        anchor_genres = []
        try:
            conn_a = get_db()
            arow = conn_a.execute(
                "SELECT genres FROM movies WHERE movie_id = ?", (int(movie_id),)
            ).fetchone()
            conn_a.close()
            if arow and arow["genres"]:
                anchor_genres = [
                    g.strip() for g in str(arow["genres"]).split("|") if g.strip()
                ]
        except Exception:
            pass

        n_latest = 0
        if include_latest:
            results, n_latest = _merge_latest_into_recommendations(
                results,
                top_n,
                {int(movie_id)},
                era_filter,
                min_avg_f,
                exclude_tmdb_supplements_from_latest=not include_regional_indian,
            )
        else:
            results = (results or [])[:top_n]

        n_regional = 0
        if include_regional_indian:
            results, n_regional = _merge_regional_into_recommendations(
                results,
                top_n,
                {int(movie_id)},
                era_filter,
                min_avg_f,
                user_profile=user_profile,
                anchor_genres=anchor_genres,
            )

        # On ML-1M, if era chips are "Modern/00s" the hybrid engine returns nothing
        # (ML-1M ends ~2000).  Fill remaining slots with TMDb popular movies
        # matched to the user's preferred genres + anchor genres.
        n_popular = 0
        if not Config.DATASET_USE_25M and era_filter and len(results) < top_n:
            results, n_popular = _merge_tmdb_popular_into_recommendations(
                results,
                top_n,
                {int(movie_id)},
                era_filter,
                min_avg_f,
                user_profile=user_profile,
                anchor_genres=anchor_genres,
            )

        rec_meta = dict(rec_meta or {})
        rec_meta["include_latest"] = bool(include_latest)
        if include_latest:
            rec_meta["included_latest"] = n_latest
        rec_meta["include_regional_indian"] = bool(include_regional_indian)
        if include_regional_indian:
            rec_meta["included_regional"] = n_regional
        if n_popular:
            rec_meta["included_tmdb_popular"] = n_popular

        for r in results or []:
            for key in (
                "hybrid_score", "content_score", "collab_score", "persona_score"
            ):
                v = r.get(key)
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    r[key] = 0.0

        # TMDB: use cache when present; otherwise fetch + write DB (same as movie detail)
        conn = get_db()
        to_enrich = []
        for r in results:
            row = conn.execute(
                "SELECT tmdb_poster_url, tmdb_overview FROM movies WHERE movie_id = ?",
                (r['movie_id'],)
            ).fetchone()
            if row and row['tmdb_poster_url']:
                r['tmdb_poster_url'] = row['tmdb_poster_url']
                r['tmdb_overview'] = row['tmdb_overview'] or ''
            else:
                to_enrich.append(r)
        conn.close()
        for r in to_enrich:
            enrich_movie_dict(r)

        payload = {'movies': results, 'meta': rec_meta or {}}
        payload = _sanitize_for_json(payload)
        cache.set(cache_key, payload, timeout=Config.CACHE_DEFAULT_TIMEOUT)
        return jsonify(payload)
    except Exception:
        current_app.logger.exception("api_recommend failed")
        return (
            jsonify(
                {
                    "error": (
                        "Could not build recommendations. "
                        "Try other era filters, a different anchor, or turn off the minimum rating. "
                        "The server log has the technical details."
                    )
                }
            ),
            500,
        )


# ──────────────────────────────────────────────
# Error handlers
# ──────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('404.html'), 500


# ──────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
