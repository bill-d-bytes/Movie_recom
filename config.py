import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Load .env from project root (works when Gunicorn's cwd is not the project dir)
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cinematch-dev-secret-change-in-prod')
    DATABASE_PATH = os.path.join(BASE_DIR, 'cinematch.db')

    # Dataset: "1m" (default) or "25m" — set MOVIELENS_DATASET=25m
    DATASET_USE_25M = os.environ.get('MOVIELENS_DATASET', '').strip().lower() in (
        '25m', '25', 'ml-25m', 'ml25m',
    )
    DATASET_DIR = os.path.join(
        BASE_DIR, 'ml-25m' if DATASET_USE_25M else 'ml-1m',
    )
    MOVIES_DAT = os.path.join(DATASET_DIR, 'movies.dat')
    RATINGS_DAT = os.path.join(DATASET_DIR, 'ratings.dat')
    USERS_DAT = os.path.join(DATASET_DIR, 'users.dat')
    MOVIES_CSV = os.path.join(DATASET_DIR, 'movies.csv')
    RATINGS_CSV = os.path.join(DATASET_DIR, 'ratings.csv')
    ML_MOVIE_SOURCE = 'ml25m' if DATASET_USE_25M else 'ml1m'

    # ML model paths
    MODELS_DIR        = os.path.join(BASE_DIR, 'models')
    TFIDF_PATH        = os.path.join(MODELS_DIR, 'tfidf_matrix.pkl')
    COSINE_SIM_PATH   = os.path.join(MODELS_DIR, 'cosine_sim.pkl')
    MOVIE_IDX_PATH    = os.path.join(MODELS_DIR, 'movie_indices.pkl')
    SVD_MODEL_PATH    = os.path.join(MODELS_DIR, 'svd_model.pkl')
    USER_ITEM_PATH    = os.path.join(MODELS_DIR, 'user_item_matrix.pkl')
    PREDICTED_PATH    = os.path.join(MODELS_DIR, 'predicted_ratings.pkl')
    # Sparse collab (no full predicted_df); written by recommender.py --build
    COLLAB_MAPPINGS_PATH = os.path.join(MODELS_DIR, 'collab_mappings.pkl')

    # TMDB API
    TMDB_API_KEY    = os.environ.get('TMDB_API_KEY', '')
    TMDB_BASE_URL   = 'https://api.themoviedb.org/3'
    TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'
    # Hybrid discover: /discover/movie from this year upward (ml-1m ends ~2000)
    TMDB_MODERN_MIN_YEAR = int(os.environ.get('TMDB_MODERN_MIN_YEAR', '2000'))

    # Supplement rows (TMDb imports) get synthetic movie_id >= this; ml-1m ids stay below
    SYNTHETIC_MOVIE_ID_BASE = 5_000_000

    # Caching
    CACHE_TYPE              = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT   = 300   # 5 minutes

    # Recommendation defaults
    TOP_N_DEFAULT = 10
    TOP_N_MAX     = 20
    COLD_START_THRESHOLD = 5  # ratings below this → content-only fallback
    # Blend newest-by-year catalog rows into /api/recommend (hybrid)
    LATEST_IN_RECOMMENDATION = 3
    LATEST_MIN_YEAR = 1990
    # TMDb discover by original_language (hi/te/ta) — Bollywood, Telugu, Tamil
    REGIONAL_IN_RECOMMENDATION = 5
    REGIONAL_MIN_YEAR = 2000
    REGIONAL_ORIGINAL_LANGUAGES = ("hi", "te", "ta")

    # Session cookies (Lax + HttpOnly; set SESSION_COOKIE_SECURE=1 behind HTTPS in production)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', '').lower() in (
        '1', 'true', 'yes', 'on',
    )


def ml_models_are_built():
    """True when TF-IDF + indices exist and collab artifacts are present (sparse or legacy)."""
    if not os.path.exists(Config.TFIDF_PATH) or not os.path.exists(Config.MOVIE_IDX_PATH):
        return False
    return os.path.exists(Config.COLLAB_MAPPINGS_PATH) or os.path.exists(
        Config.PREDICTED_PATH
    )
