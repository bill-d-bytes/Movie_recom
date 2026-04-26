# CineMatch

Hybrid movie recommender (MovieLens 1M + optional TMDB posters), built with **Flask**, **scikit-learn**, and **Jinja2** + **Tailwind** front end.

## Prerequisites

- Python 3.9+ (3.10+ recommended if you need `hashlib.scrypt`; the app uses **pbkdf2** for passwords for portability)
- `ml-1m/` dataset files (`movies.dat`, `ratings.dat`, `users.dat`)
- (Optional) **TMDB API key** for posters and overviews: [The Movie DB](https://www.themoviedb.org/settings/api)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set SECRET_KEY and optionally TMDB_API_KEY
```

### Database and ML models

```bash
# One-time: build recommendation artefacts into models/
python recommender.py --build
```

The app will create/seed `cinematch.db` on first import if needed (MovieLens + ratings).

## Run (development)

```bash
source venv/bin/activate
python app.py
# Open http://127.0.0.1:5000
```

## Run (production-style)

```bash
pip install gunicorn
export PORT=5000
gunicorn -w 2 -b 0.0.0.0:$PORT app:app
# Or: heroku local / Procfile
```

Set `SESSION_COOKIE_SECURE=1` in production when serving over **HTTPS**.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
# Unit only: pytest tests/ -m unit -q
# Integration: pytest tests/ -m integration -q
```

See `Details/TEST_PLAN.md` for coverage by phase.

## Optional: slow API logging

To log `/api/*` calls slower than 2000 ms (milliseconds):

```bash
export CINEMATCH_LOG_SLOW_API_MS=2000
python app.py
```

## Project layout (high level)

| Path | Purpose |
|------|---------|
| `app.py` | Flask app, routes, session auth |
| `recommender.py` | Build/load models, `hybrid_recommend` |
| `database.py` | SQLite, MovieLens seeding, queries |
| `tmdb.py` | Poster/overview fetch + cache in DB |
| `app/templates/` | Jinja pages |
| `app/static/js/api.js` | `apiFetch`, toasts, cards, autocomplete |
| `models/*.pkl` | Trained artefacts (from `recommender.py --build`) |
| `Details/` | Status, test plan, design notes |

## Documentation

- `Details/CURRENT_STATE.md` — delivery status and remaining phases
- `Details/TEST_PLAN.md` — how tests map to product phases
- `implementation_plan.md` — original phased plan (partly historical)
