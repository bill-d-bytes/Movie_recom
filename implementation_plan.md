# CineMatch – Smart Movie Recommender System
## Phased Implementation Plan *(Updated)*

> Derived from: **PRD** · **DRD** · **ml-1m dataset** (already on disk) · **Frontend HTML screens** (already built)

---

## ✅ What Already Exists

### Dataset — `ml-1m/` (MovieLens 1M)
| File | Format | Records |
|---|---|---|
| `movies.dat` | `MovieID::Title::Genres` (pipe-separated genres) | 3,883 movies |
| `ratings.dat` | `UserID::MovieID::Rating::Timestamp` | 1,000,209 ratings |
| `users.dat` | `UserID::Gender::Age::Occupation::Zip-code` | 6,040 users |

**Key data notes:**
- Delimiter is `::` (not comma) → use `pd.read_csv(..., sep='::', engine='python')`
- Age is coded (1=Under 18, 18=18-24, 25=25-34, 35=35-44, 45=45-49, 50=50-55, 56=56+)
- Gender: `M` / `F`
- Occupation: 21 coded categories (0–20)
- Genres: pipe-separated (e.g. `Action|Adventure|Thriller`)
- Each user has **at least 20 ratings** → no ultra-sparse cold-start within dataset
- No IMDB IDs in dataset → need TMDB API to fetch posters by title+year

### Frontend — (historical) ~~`frontend/` static mocks~~ **removed**; use **`app/templates/`**
**Note:** The old `frontend/*/code.html` design exports were removed from the project; the live Jinja2 app is the only UI copy.
- **CSS Framework:** Tailwind CSS (CDN, with forms + container-queries plugins)
- **Font:** `Be Vietnam Pro` (400 / 500 / 700 / 800 / 900)
- **Icons:** Google Material Symbols Outlined
- **Theme:** Deep dark (#131313 bg), Cinematic red (#e50914) as primary-container / CTA
- **Style:** Glassmorphism overlays, backdrop-blur, tonal layering
- **Layout:** Fixed-fluid 12-col grid, max-width 1440px, 4px baseline grid, 20px gutters

| Screen Directory | HTML File | Status | Screen |
|---|---|---|---|
| `login_cinematch/` | `code.html` | ✅ Built | Sign In form, glassmorphic card, cinematic BG |
| `profile_cinematch/` | `code.html` | ✅ Built | User info form (name, email, location, DOB) + avatar, notifications toggle |
| `preferences_cinematch/` | `code.html` | ✅ Built | Genre chips (multi-select), age input, gender select |
| `filters_cinematch/` | `code.html` | ✅ Built | Text search, era checkboxes (Classic/80s/90s/00s/Modern), more filters |
| `discover_cinematch/` | `code.html` | ✅ Built | Main discovery/recommendations grid page |
| `movie_details_cinematch/` | `code.html` | ✅ Built | Movie detail modal/page with scores display |
| `cinematic_noir/` | `DESIGN.md` | ✅ Design doc only | Tokens, typography, spacing, component specs |

> **None of the HTML files are yet wired to a Flask backend.** All forms have `action="#"`, no `fetch()` calls, no dynamic data. All content is static placeholder data.

---

## Project Overview

**Product Name:** CineMatch – Movie Recommendation System
**Type:** Web-based, ML-powered, Three-Tier Architecture
**Stack:**
- **Backend:** Python 3.x + Flask
- **ML:** Scikit-learn · Pandas · NumPy · Scipy
- **Frontend:** HTML + Tailwind CSS (CDN) + Vanilla JS (fetch API) — *screens already built*
- **Database:** SQLite (dev) / MySQL (prod-ready)
- **Data:** MovieLens 1M (`ml-1m/`) — *already on disk*
- **External API:** TMDB API (posters + movie metadata)
- **Auth:** Werkzeug password hashing + Flask sessions

**Core Hybrid Formula:**
```
Hybrid Score = 0.5 × Content Score + 0.3 × Collaborative Score + 0.2 × Personalization Score
```

---

## Phase 1 — Foundation & Project Setup
**Duration:** ~3–4 days *(faster — dataset already on disk)*
**Goal:** Set up Flask skeleton, database schema, and parse the existing ml-1m dataset.

### Tasks
- [ ] Create Flask project structure:
  ```
  /app
    /templates      ← move/copy HTML from frontend/ dirs here
    /static
      /css           ← any local overrides
      /js            ← main.js, api.js
    /models          ← .pkl files
    /data            ← ml-1m/ symlink or copy
  app.py
  requirements.txt
  config.py
  ```
- [ ] Install dependencies:
  ```
  Flask, pandas, numpy, scikit-learn, scipy, requests, werkzeug, flask-caching
  ```
- [ ] **Parse ml-1m dataset** (separator is `::`!):
  ```python
  movies  = pd.read_csv('ml-1m/movies.dat',  sep='::', engine='python',
                        names=['movie_id','title','genres'], encoding='latin-1')
  ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                        names=['user_id','movie_id','rating','timestamp'])
  users   = pd.read_csv('ml-1m/users.dat',   sep='::', engine='python',
                        names=['user_id','gender','age','occupation','zip'])
  ```
- [ ] Parse release year from movie title: `"Toy Story (1995)"` → year = 1995
- [ ] Explode genres into list column for TF-IDF and filtering
- [ ] Set up SQLite DB with schema:
  - **app_users** → `user_id` (PK), `username`, `email`, `password_hash`, `age`, `gender`, `preferred_genres` (JSON list)
  - **movies** → `movie_id`, `title`, `genres`, `year`, `tmdb_poster_url`, `tmdb_overview`
  - **ratings** → `user_id`, `movie_id`, `rating` (from ml-1m)
- [ ] Seed movies table from `movies.dat`
- [ ] Seed ratings table from `ratings.dat`
- [ ] Obtain **TMDB API key** → test poster fetch by title+year (use `/search/movie`)
- [ ] Set up Flask app skeleton with Blueprint routing

### Deliverables
- Flask app running at `localhost:5000`
- SQLite DB seeded with 3,883 movies + 1M ratings
- ml-1m DataFrames loading correctly
- TMDB test call returning a poster URL

---

## Phase 2 — ML Engine (Core Feature)
**Duration:** ~1.5–2 weeks
**Goal:** Build, train, test, and persist all three recommendation engines using ml-1m data.

### 2A · Content-Based Filtering
- [ ] Build feature string per movie: `"Toy Story Animation Children's Comedy 1995"`
- [ ] Apply **TF-IDF vectorization** (`TfidfVectorizer`, max_features=5000)
- [ ] Compute **cosine similarity matrix** (sparse, ~3900×3900)
- [ ] `get_content_recommendations(movie_id, top_n=10)` → ranked list of similar movie IDs
- [ ] Save: `models/tfidf_matrix.pkl`, `models/cosine_sim.pkl`, `models/movie_indices.pkl`

### 2B · Collaborative Filtering (SVD)
- [ ] Build **User–Item matrix**: 6040 users × 3883 movies (sparse)
- [ ] Apply **Truncated SVD** (`TruncatedSVD`, n_components=50)
- [ ] Reconstruct predicted ratings matrix
- [ ] `get_collab_recommendations(user_id, top_n=10)` → ranked list of predicted movie IDs
- [ ] Evaluate: RMSE on 20% held-out split → target < 1.8
- [ ] Save: `models/svd_model.pkl`, `models/user_item_matrix.pkl`

### 2C · Personalization Scoring
- [ ] Map ml-1m age codes to age groups matching filters UI (Classic/80s/90s/00s/Modern)
  - Age 1 (under 18) → Modern/2000s bias
  - Age 25–35 → 90s/00s bias, etc.
- [ ] Gender-based genre affinity (from aggregated rating patterns in `ratings.dat`)
- [ ] Genre preference score: overlap between user's `preferred_genres` and movie genres (Jaccard)
- [ ] Normalize all sub-scores to [0, 1] before combining

### 2D · Hybrid Engine
- [ ] `hybrid_recommend(user_id, movie_id, user_profile, top_n=10)`:
  ```
  score = 0.5×content + 0.3×collaborative + 0.2×personalization
  ```
- [ ] Cold-start fallback: if user has < 5 ratings → use content-based only (weight: 0.8 content + 0.2 personalization)
- [ ] Apply era/year filter if user selected eras on Filters screen
- [ ] Return: list of dicts with `movie_id`, `title`, `genres`, `hybrid_score`, `content_score`, `collab_score`, `persona_score`

### Deliverables
- All `.pkl` model files saved to `models/`
- `recommender.py` module with clean public API
- Evaluation notebook: RMSE, Precision@10, Recall@10
- Cold-start path tested

---

## Phase 3 — Backend API (Flask)
**Duration:** ~1 week
**Goal:** Build all Flask routes that the existing frontend HTML screens will call.

### 3A · Auth Routes
| Method | Route | Description |
|---|---|---|
| `POST` | `/auth/register` | Create user, hash pw (Werkzeug), save to DB |
| `POST` | `/auth/login` | Validate credentials, start session |
| `POST` | `/auth/logout` | Clear session |

### 3B · Profile Routes
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/profile` | Return current user's profile JSON |
| `PUT` | `/api/profile` | Update name, email, age, gender, location |

### 3C · Preferences Routes
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/preferences` | Return saved genre preferences |
| `PUT` | `/api/preferences` | Save genre list + age + gender from Preferences screen |

### 3D · Movie Routes
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/movies/search?q=` | Autocomplete: title search on local movies table |
| `GET` | `/api/movies/trending` | Top-rated movies from ratings.dat (avg rating, min 50 reviews) |
| `GET` | `/api/movies/<movie_id>` | Full detail: title, genres, year, poster (TMDB), overview, scores |

### 3E · Recommendation Route
| Method | Route | Description |
|---|---|---|
| `POST` | `/api/recommend` | Body: `{movie_id, era_filter[], sort_by}` → run hybrid engine → return top-N list |

### 3F · Jinja2 Page Routes (serve HTML templates)
| Route | Template | Screen |
|---|---|---|
| `GET /` | `login_cinematch/code.html` | Login (unauthenticated landing) |
| `GET /discover` | `discover_cinematch/code.html` | Discover / Recommendations |
| `GET /profile` | `profile_cinematch/code.html` | Profile |
| `GET /preferences` | `preferences_cinematch/code.html` | Preferences |
| `GET /filters` | `filters_cinematch/code.html` | Filters |
| `GET /movie/<id>` | `movie_details_cinematch/code.html` | Movie Detail |

### 3G · Performance & Security
- [ ] Pre-load all `.pkl` models at Flask `app.before_first_request`
- [ ] Cache TMDB poster URLs in DB to avoid repeat API calls
- [ ] Flask-Caching for recommendation results (TTL = 300s)
- [ ] `@login_required` decorator on all `/api/*` and page routes except `/` and `/auth/*`
- [ ] Input validation + error JSON responses
- [ ] Response time target: < 2 seconds end-to-end

### Deliverables
- All routes implemented and tested with Postman / curl
- Session auth working across all protected routes
- TMDB poster URLs cached in DB

---

## Phase 4 — Frontend Integration (Wire HTML → Flask)
**Duration:** ~1 week *(reduced — screens already built, just needs wiring)*
**Goal:** Convert static HTML screens into dynamic Flask templates connected to the backend API.

> **The HTML, design, and layout are already complete.** This phase is purely about integration — replacing static placeholder data with real API calls and making forms functional.

### 4A · Convert to Jinja2 Templates
- [ ] Copy each `frontend/*/code.html` → `app/templates/{screen_name}.html`
- [ ] Add Jinja2 template inheritance: create `base.html` with shared `<head>`, nav, scripts
- [ ] Extend other templates from `base.html`
- [ ] Inject Flask CSRF token into all `<form>` tags

### 4B · Login Screen (`login_cinematch`)
- [ ] Change `<form action="#">` → `<form action="/auth/login" method="POST">`
- [ ] Add register link → `/auth/register` (create simple register page or modal)
- [ ] On success: redirect to `/discover`; on failure: show error message via Jinja flash

### 4C · Profile Screen (`profile_cinematch`)
- [ ] On page load: `fetch('/api/profile')` → populate name, email, date fields
- [ ] "Save Changes" button: `PUT /api/profile` with form data → show success toast
- [ ] Wire notifications toggle to user preference save

### 4D · Preferences Screen (`preferences_cinematch`)
- [ ] Genre chips: read active state from `GET /api/preferences` on load
- [ ] Age + gender: pre-fill from profile
- [ ] "Continue" / save button: `PUT /api/preferences` → redirect to `/filters` or `/discover`

### 4E · Filters Screen (`filters_cinematch`)
- [ ] Text search box: `GET /api/movies/search?q=` → autocomplete dropdown (debounced 300ms)
- [ ] Era checkboxes: tracked in JS state (Classic / 80s / 90s / 00s / Modern)
- [ ] "Discover" button: `POST /api/recommend` with `{movie_id, era_filter[]}` → navigate to `/discover` with results in sessionStorage or URL params

### 4F · Discover Screen (`discover_cinematch`)
- [ ] On load: call `POST /api/recommend` (or read from sessionStorage) → render movie cards dynamically
- [ ] Each card: insert real poster URL, title, genre tags, hybrid score
- [ ] Card hover → show "Match %" (hybrid_score × 100)
- [ ] Card click → navigate to `/movie/{movie_id}`
- [ ] Loading spinner while API call is in flight

### 4G · Movie Detail Screen (`movie_details_cinematch`)
- [ ] On load: `GET /api/movies/{movie_id}` → populate poster, title, genres, overview, year
- [ ] Show 3 score bars: Content Score / Collaborative Score / Personalization Score
- [ ] "More Like This" section: `POST /api/recommend` with same movie → show 4–6 cards

### 4H · Shared JS (`static/js/api.js`)
- [ ] Centralised `apiFetch(url, method, body)` wrapper with auth error handling (redirect to `/` on 401)
- [ ] Toast notification component (reuse on all screens)
- [ ] Loading spinner utility

### Deliverables
- All 6 screens fully dynamic with real data from Flask
- Forms submitting to correct routes
- No `action="#"` remaining — all wired
- Static placeholder text replaced with live DB/API data

---

## Phase 5 — Testing, Polish & Deployment
**Duration:** ~1 week
**Goal:** End-to-end testing, performance validation, polish, and optional deployment.

### Testing
- [ ] End-to-end walkthrough: Register → Login → Preferences → Filters → Discover → Movie Detail
- [ ] Cold-start user path (0 prior ratings)
- [ ] Edge cases: no search results, invalid login, TMDB API timeout, movie with no poster
- [ ] RMSE < 1.8 confirmed on held-out test split
- [ ] Response time < 2s on recommend endpoint (models pre-loaded)

### Polish
- [ ] All error states handled (form validation flash messages, 404 page)
- [ ] Card hover animations from DESIGN.md: `scale-105` + red glow
- [ ] Loading skeleton UI on Discover grid while recommendations load
- [ ] Favicon + correct `<title>` tags per page
- [ ] `favicon.ico` referencing CineMatch brand mark

### Deployment (Optional)
- [ ] `gunicorn` for production WSGI
- [ ] Environment variables: `TMDB_API_KEY`, `SECRET_KEY`, `DATABASE_URL`
- [ ] Deployable to Render / Railway / Heroku / local server
- [ ] Optional: Docker + `docker-compose.yml`

### Documentation
- [ ] `README.md`: setup, install, run instructions
- [ ] API route documentation (can use Flask-RESTX or simple markdown)
- [ ] ML evaluation results summary

### Deliverables
- Fully working CineMatch web app
- Saved model files in `models/`
- Complete documentation

---

## Revised Timeline

| Phase | Focus | Original | **Revised** |
|---|---|---|---|
| Phase 1 | Foundation & Setup | 1 week | **3–4 days** *(dataset exists)* |
| Phase 2 | ML Engine | 2 weeks | **2 weeks** |
| Phase 3 | Flask Backend API | 1–1.5 weeks | **1 week** |
| Phase 4 | Frontend Integration | 1.5–2 weeks | **1 week** *(screens built)* |
| Phase 5 | Testing, Polish, Deploy | 1 week | **1 week** |
| **Total** | | **~6.5–7.5 weeks** | **~5.5–6 weeks** |

> **~1–2 weeks saved** because the ml-1m dataset is already downloaded and all 7 frontend screens (Login, Profile, Preferences, Filters, Discover, Movie Details + Design system) are fully coded in Tailwind CSS.

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Cold-start for new app users | Low personalization | Content-based fallback (weight 0.8/0.2) |
| ml-1m uses `::` separator | Parse errors | Use `engine='python'`, `encoding='latin-1'` |
| TMDB rate limits | No posters | Cache URLs in DB; add fallback placeholder image |
| Large cosine similarity matrix (~3900²) | Memory | Use sparse matrix; truncate to top-50 per movie at build time |
| ml-1m has no IMDB/TMDB IDs | Poster lookup difficulty | Match by `Title (Year)` → TMDB `/search/movie?query=Title&year=Year` |

---

## Future Enhancements (Post-MVP)
- Deep Learning recommender (neural collaborative filtering / two-tower model)
- Sentiment analysis on user reviews
- Real-time recommendation updates (WebSockets)
- Mobile app (React Native / Flutter)
- OTT platform integration (Netflix/Prime/Hotstar links)
