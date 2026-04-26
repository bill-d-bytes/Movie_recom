# CineMatch — Test plan

> Maps **phases** (from `implementation_plan.md` and `Details/CURRENT_STATE.md`) to **features**, then to **test types** (unit vs integration) and **how to run** them in `tests/`.

---

## Conventions

| Type | Definition |
|------|------------|
| **Unit** | Isolated logic, no live Flask server, or Flask route with mocks / minimal I/O. Fast. |
| **Integration** | Flask `test_client`, real SQLite at `cinematch.db` (or env `CINEMATCH_DATABASE_PATH`), optional ML **`.pkl`** files. Exercises multiple components. |
| **E2E (manual / future)** | Browser (Playwright/Cypress) — out of scope for this repo; listed as future work. |

### Environment notes (repo-specific)

- **Password hashing** — Registration uses `generate_password_hash(..., method="pbkdf2:sha256")` so it runs on Python builds where `hashlib.scrypt` is unavailable (Werkzeug’s default is scrypt).
- **SQLite** — `database.get_db()` uses a **30s connection timeout** to reduce `database is locked` during tests and concurrent access (WAL mode).
- **Persistent DB** — Integration tests insert real rows into `cinematch.db`. Tests use **UUID-based emails/usernames** to avoid unique-constraint flakiness across runs.
- **ML** — `POST /api/recommend` returns **503** if `models/cosine_sim.pkl` (etc.) is missing; tests accept either 200+JSON list or 503.
- **Slow API logging** — Set `CINEMATCH_LOG_SLOW_API_MS=2000` to log `/api/*` responses over that duration (see `app.py`). **Production** — set `SESSION_COOKIE_SECURE=1` when using HTTPS (see `config.py`, `.env.example`).

**Markers (pytest):**

- `@pytest.mark.unit` — no DB, or only pure helpers
- `@pytest.mark.integration` — app + database (+ ML when present)

**Run commands:**

```bash
pip install -r requirements-dev.txt
cd /path/to/Movie_recom
pytest tests/ -m unit -q                    # unit only
pytest tests/ -m integration -q             # integration (needs DB; ML optional)
pytest tests/ -q                            # all
```

---

## Phase 1 — Foundation & project setup

| Feature | Unit tests | Integration tests |
|---------|------------|-------------------|
| Config & paths | — | `test_integration_app.py` — `GET /` returns 200 when not logged in |
| SQLite schema (`create_tables`) | — | Implicit via any DB read in integration tests |
| **Dataset parsing** (year from title) | `test_unit_database` — `_extract_year` / title patterns | `search_movies` / `get_movie_by_id` (integration) |
| **Seed** (`movies`, `ratings`) | — | Covered by use of `cinematch.db` with data |

---

## Phase 2 — ML engine

| Feature | Unit tests | Integration tests |
|---------|------------|-------------------|
| **TF-IDF / cosine** (artefacts) | — | `test_integration_recommend` — if models exist, `POST /api/recommend` returns 200 + JSON list |
| **SVD / hybrid weights** | Optional: `recommender.ERA_*` invariants in `test_unit_recommender` | Same as above; **503** if `cosine_sim.pkl` missing (assert error JSON) |
| **Cold-start** / personalization | — | New user: `recommend` still returns 200 with models loaded |

---

## Phase 3 — Backend API (Flask)

| Feature | Unit tests | Integration tests |
|---------|------------|-------------------|
| **Register** | — | `test_api_auth` — 201, 400, 409 |
| **Login / logout** | `test_unit_werkzeug` — hash verify | `test_api_auth` — 200, 401; session after logout |
| **`GET/PUT /api/profile`** | — | `test_api_profile` — 401 unauthenticated, 200 GET, 200 PUT |
| **`GET/PUT /api/preferences`** | — | `test_api_preferences` — 200 GET/PUT |
| **`GET /api/movies/search`** | `search` short query → `[]` (integration) | `test_api_movies` — 200, list shape with auth |
| **`GET /api/movies/trending`** | — | `test_api_movies` — 200 with session |
| **`GET /api/movies/<id>`** | — | `test_api_movies` — 200 for `movie_id=1`, 404 missing |
| **`POST /api/recommend`** | — | `test_integration_recommend` — 400 without `movie_id` |
| **401 on `/api/*`** | — | `test_api_profile` / `test_api_movies` |

---

## Phase 4 — Frontend integration (templates + JS)

| Feature | Unit tests | Integration tests |
|---------|------------|-------------------|
| **Jinja** login / register | — | `test_integration_app.py` — `GET /auth/register` contains “Create your account” |
| **Pages** (discover, profile, …) | — | Authenticated `GET` returns 200 for `/discover`, `/filters` |
| **sessionStorage** discover flow | E2E / manual | Documented: apply filters in browser → discover shows matches |

---

## Phase 5 — Testing, polish, deployment

| Feature | Unit tests | Integration tests |
|---------|------------|-------------------|
| **404 / 500 JSON** for `/api/*` | — | Optional: request bogus `/api/…` (future) |
| **Performance** (&lt; 2s recommend) | — | Manual or CI timing (future `pytest` mark `slow`) |
| **README / deploy** | — | Smoke: `flask` / `gunicorn` (manual) |

---

## Traceability matrix (pytest modules)

| Module | Type | Phases covered |
|--------|------|----------------|
| `tests/test_unit_database.py` | unit | 1 |
| `tests/test_unit_recommender.py` | unit | 2 |
| `tests/test_unit_werkzeug.py` | unit | 3 (auth crypto) |
| `tests/test_api_auth.py` | integration | 3 |
| `tests/test_api_profile.py` | integration | 3 |
| `tests/test_api_preferences.py` | integration | 3 |
| `tests/test_api_movies.py` | integration | 1, 3 |
| `tests/test_integration_recommend.py` | integration | 2, 3 |
| `tests/test_integration_app.py` | integration | 1, 3, 4 |

---

## After integration: what to run

1. **After a backend or DB change** — `pytest tests/ -m integration`
2. **After ML / `recommender.py` change** — `pytest tests/ -m integration` and confirm `POST /api/recommend` (not 503)
3. **CI suggestion** — cache `cinematch.db` + `models/*.pkl` or mark ML-dependent tests with `@pytest.mark.skipif(not models_exist, …)`

---

## Related documents

- `Details/CURRENT_STATE.md` — delivery status and pending phases.
- `implementation_plan.md` — original phased implementation.

---

## Future work

- **E2E** (Playwright): register → preferences → filters → discover → movie detail
- **Load tests** (Locust) for `/api/recommend`
- **Contract tests** for TMDB (mock `requests`)
