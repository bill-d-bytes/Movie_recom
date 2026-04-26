# CineMatch — Project Status

> Last reviewed: **2026-04-25**. **Phase A** done. **Phase B** (quality / ops) largely done: see **Phase B** section below. **Tests**: `Details/TEST_PLAN.md`, `tests/`.

---

## Current state

### Product and stack

| Item | Detail |
|------|--------|
| **Name** | CineMatch — hybrid movie recommender (MovieLens 1M + TMDB) |
| **Backend** | Python 3.x, Flask 3.x, Werkzeug sessions, Flask-Caching |
| **ML** | scikit-learn (TF-IDF, cosine similarity, Truncated SVD), pandas, numpy, scipy |
| **Frontend (live)** | Jinja2 templates + Tailwind CDN + vanilla JS (`app/templates/`, `app/static/js/api.js`) |
| **Database** | SQLite (`cinematch.db`); schema and seeding via `database.py`; connections use **30s timeout** (WAL-friendly, test-safe) |
| **Config** | `config.py` + `.env` (`python-dotenv`); see **`.env.example`**. **Session cookies**: HttpOnly, `SameSite=Lax`; set **`SESSION_COOKIE_SECURE=1`** behind HTTPS in production. |
| **Ops** | Optional **`CINEMATCH_LOG_SLOW_API_MS`** (e.g. `2000`) logs slow `/api/*` calls. **Procfile** + **`gunicorn`** in `requirements.txt` for deploy. |
| **Design reference** | **Removed** (2026): old static HTML mocks under `frontend/`; live UI is **`app/templates/`** only. |

### Data and models

- **MovieLens 1M** in `ml-1m/` (`movies.dat`, `ratings.dat`, `users.dat`) — `::` delimited, latin-1.
- **Trained artifacts** in `models/` (present on disk):
  - `tfidf_matrix.pkl`, `cosine_sim.pkl`, `movie_indices.pkl`
  - `svd_model.pkl`, `user_item_matrix.pkl`, `predicted_ratings.pkl`
- **Recommendation engine** in `recommender.py`: content + collaborative + hybrid (weights 0.5 / 0.3 / 0.2) with personalization and era filters; build via `python recommender.py --build`.
- **TMDB** integration in `tmdb.py` for poster/metadata enrichment (with fallback `app/static/img/no_poster.png`).

### Application entrypoint

- **`app.py`**: creates tables, seeds from MovieLens if needed, calls `engine.load()` when cosine sim model path exists, registers routes and `login_required` for protected pages and `/api/*`.
- **Password hashing**: registration uses **`pbkdf2:sha256`** (`generate_password_hash(..., method="pbkdf2:sha256")`) so auth works on Python builds without **`hashlib.scrypt`** (Werkzeug’s default would use scrypt).

### Implemented behavior (high level)

- **Auth**: `POST /auth/login`, `POST /auth/logout`, `POST /auth/register` (JSON), `GET /auth/register` — **`login.html` uses Jinja `mode == 'register'`** for email + username + password sign-up.
- **Pages**: `/`, `/discover`, `/profile`, `/preferences`, `/filters`, `/movie/<id>` (Jinja templates; protected except `/` and auth).
- **APIs**: `GET/PUT /api/profile`, `GET/PUT /api/preferences`, `GET /api/movies/search`, `GET /api/movies/trending`, `GET /api/movies/<id>`, `POST /api/recommend` (requires `movie_id`, optional `era_filter`, `top_n`).
- **Client wiring**: `app/static/js/api.js` — `apiFetch`, **`parseApiError`** (JSON + 503 message), toasts, spinner, `buildMovieCard`, `initMovieAutocomplete` (empty search → “No movies match”). Used on login, profile, preferences, discover, movie detail, filters.
- **Error / empty states (Phase B)**: Discover (trending + parse failures), movie detail (404, recommend 503/empty), filters (recommend errors) show clearer copy; TMDB: placeholder poster + note when no overview.

### Screen-level notes

| Screen | Status |
|--------|--------|
| Login / Register | Login uses `apiFetch` → `/auth/login`. Register (same template, `mode=register`) → `POST /auth/register`; “Sign In” / “Sign Up” cross-links. |
| Profile / Preferences | Load/save via API; wired. |
| Discover | If **`sessionStorage` key `cm_discover_recommendations`** is set (one-shot) after **Filters → Save**, shows **hybrid** list; otherwise loads **`/api/movies/trending`**. |
| Movie detail | Loads movie via API; “more like this” uses **`POST /api/recommend`**. |
| Filters | **Autocomplete** (`initMovieAutocomplete`) + anchor hint; **Save** → `POST /api/recommend` with `movie_id` + `era_filter` → `sessionStorage` → `/discover`. Min. rating UI is still decorative (not sent to API). |
| Nav (mobile) | Bottom nav in discover / profile / preferences / filters uses **`/discover`**, **`/filters`**, **`/profile`**; preferences bottom bar includes **Prefs** → `/preferences`. `login.html` “Forgot password” remains `#` (not implemented). |

### Testing (automated)

- **`Details/TEST_PLAN.md`** — phases → features → unit vs integration matrix, run commands, environment notes.
- **`tests/`** — pytest: **unit** (`_extract_year`, recommender constants, password hash) and **integration** (Flask client, auth, profile, preferences, movies, recommend, pages). Markers: `@pytest.mark.unit`, `@pytest.mark.integration`.
- **Dev deps**: `requirements-dev.txt` (pytest, pytest-cov). **`pytest.ini`** at repo root.
- **Run**: `pip install -r requirements-dev.txt` then `pytest tests/ -q` (or `-m unit` / `-m integration`). Integration tests use the real **`cinematch.db`** with UUID-based users to avoid clashes.

### Dependencies

- **Runtime**: `requirements.txt` (Flask, pandas, numpy, scikit-learn, scipy, requests, Werkzeug, Flask-Caching, python-dotenv).
- **Dev / CI**: `requirements-dev.txt`.
- Local **`venv/`** may be present for development.

### Documentation drift

- Root **`implementation_plan.md`** still describes the frontend as unwired static HTML; the **live app** lives under **`app/`** and is largely integrated. Treat `implementation_plan.md` as a historical plan unless refreshed.

---

## Pending phases

Work below maps to the original phased plan (foundation → ML → API → UI integration → polish) but only lists what **remains** or **warrants verification**.

### Phase A — User flows and filters/discover

**Status: done in repo.**

- Optional follow-up: wire **min rating** slider to the API if the backend adds a `min_rating` (or similar) parameter.

### Phase B — Quality, security, and operations

**Status: mostly done** — session hardening, UI error/empty states, `parseApiError` + 503 copy, slow-API opt-in logging, **README.md**, **`.env.example`**, **Procfile**, **gunicorn** in `requirements.txt`.

1. **CSRF** — **Open** for future HTML form posts to the API. Current app uses **JSON** + `SameSite` session cookies; no CSRF tokens added.
2. **Errors and empty states** — **Done** in `api.js` + discover / movie detail / filters (see `parseApiError`, trending/recommend copy).
3. **Performance** — **Opt-in** via `CINEMATCH_LOG_SLOW_API_MS`. Manual/CI timing of recommend still optional; DB caches TMDB fields.
4. **Testing** — **Automated** in `tests/`. **E2E** (Playwright, etc.): still **open**.
5. **Deployment** — **Documented** (gunicorn, Procfile). **Docker** / host-specific steps: still optional.

### Phase C — Product and ML validation

1. **Metrics**: RMSE on held-out data (recommender), optional Precision@K / Recall@K; add a short “Evaluation” subsection to **README** or a separate `docs/` note.
2. **README** — **Partially done** (run, test, env, gunicorn). **Remaining**: optional ML evaluation numbers in README.

### Phase D — Cleanup (optional)

1. **`frontend/`** static HTML mocks **removed** from the repo; only **`app/templates/`** is authoritative.
2. Refresh **`implementation_plan.md`** checkboxes to match this file or mark it superseded.

---

## Related documents

- **`README.md`** — install, run, test, production notes.
- **`Details/TEST_PLAN.md`** — test matrix and how to run pytest.
- `implementation_plan.md` — phased implementation (partially outdated vs repo).
- `PRODUCT REQUIREMENTS DOCUMENT (PRD) (3).md` — product requirements.
- `Details/DESIGN REQUIREMENTS DOCUMENT (DRD) (2).md` — design and UI notes (file includes large embedded image data).
