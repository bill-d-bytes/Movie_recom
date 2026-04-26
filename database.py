"""
database.py — SQLite setup and seeding from ml-1m dataset.
Run standalone: python database.py  (seeds the DB)
"""
import sqlite3
import json
import re
import difflib
import pandas as pd
from config import Config


# ──────────────────────────────────────────────
# Connection helper
# ──────────────────────────────────────────────
def get_db():
    # timeout mitigates "database is locked" under test clients + concurrent readers (WAL)
    conn = sqlite3.connect(Config.DATABASE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ──────────────────────────────────────────────
# Schema creation
# ──────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS app_users (
    user_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE,
    email           TEXT    NOT NULL UNIQUE,
    password_hash   TEXT    NOT NULL,
    age             INTEGER,
    gender          TEXT,
    location        TEXT,
    dob             TEXT,
    preferred_genres TEXT   DEFAULT '[]',
    notifications   INTEGER DEFAULT 1,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS movies (
    movie_id        INTEGER PRIMARY KEY,
    title           TEXT    NOT NULL,
    genres          TEXT    NOT NULL,   -- pipe-separated  e.g. "Action|Comedy"
    year            INTEGER,
    tmdb_poster_url TEXT    DEFAULT '',
    tmdb_overview   TEXT    DEFAULT '',
    tmdb_id         INTEGER,             -- TMDb id for supplement rows; NULL for ml-1m
    source          TEXT    DEFAULT 'ml1m'  -- "ml1m" | "tmdb"
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id     INTEGER NOT NULL,
    movie_id    INTEGER NOT NULL,
    rating      REAL    NOT NULL,
    PRIMARY KEY (user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_user    ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie   ON ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_movies_title    ON movies(title);
"""


def create_tables(conn):
    conn.executescript(SCHEMA)
    conn.commit()
    print("✅ Tables created.")


# ──────────────────────────────────────────────
# Dataset parsing helpers
# ──────────────────────────────────────────────
def _extract_year(title: str):
    """Extract year from 'Movie Title (YYYY)' format."""
    match = re.search(r'\((\d{4})\)\s*$', title)
    return int(match.group(1)) if match else None


def load_movies_df() -> pd.DataFrame:
    df = pd.read_csv(
        Config.MOVIES_DAT, sep='::', engine='python',
        names=['movie_id', 'title', 'genres'], encoding='latin-1'
    )
    df['year'] = df['title'].apply(_extract_year)
    return df


def load_ratings_df() -> pd.DataFrame:
    return pd.read_csv(
        Config.RATINGS_DAT, sep='::', engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1'
    )


def load_users_df() -> pd.DataFrame:
    return pd.read_csv(
        Config.USERS_DAT, sep='::', engine='python',
        names=['user_id', 'gender', 'age', 'occupation', 'zip'], encoding='latin-1'
    )


# ──────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────
def seed_movies(conn, df: pd.DataFrame):
    """Insert movies from movies.dat — skip if already seeded."""
    count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    if count > 0:
        print(f"ℹ️  Movies already seeded ({count} rows). Skipping.")
        return

    rows = [
        (int(row.movie_id), row.title, row.genres,
         int(row.year) if pd.notna(row.year) else None, '', '', None, 'ml1m')
        for row in df.itertuples()
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO movies (movie_id, title, genres, year, tmdb_poster_url, tmdb_overview, tmdb_id, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()
    print(f"✅ Seeded {len(rows)} movies.")


def seed_ratings(conn, df: pd.DataFrame):
    """Insert ratings from ratings.dat — skip if already seeded."""
    count = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    if count > 0:
        print(f"ℹ️  Ratings already seeded ({count} rows). Skipping.")
        return

    print(f"   Seeding {len(df):,} ratings (this may take ~30s)…")
    chunk_size = 50_000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        rows = [(int(r.user_id), int(r.movie_id), float(r.rating))
                for r in chunk.itertuples()]
        conn.executemany(
            "INSERT OR IGNORE INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)",
            rows
        )
        conn.commit()
        print(f"   …{min(i + chunk_size, len(df)):,} / {len(df):,}", end='\r')
    print(f"\n✅ Seeded {len(df):,} ratings.")


# ──────────────────────────────────────────────
# Public query helpers (used by Flask routes)
# ──────────────────────────────────────────────
def get_movie_by_id(movie_id: int):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM movies WHERE movie_id = ?", (movie_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _norm_title(s: str) -> str:
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", (s or "")).strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
    return " ".join(t.split())


def search_movies(query: str, limit: int = 20):
    q = (query or "").strip()
    if len(q) < 2:
        return []
    conn = get_db()
    rows = conn.execute(
        "SELECT movie_id, title, genres, year FROM movies "
        "WHERE LOWER(title) LIKE LOWER(?) ORDER BY (year IS NULL) ASC, year DESC LIMIT ?",
        (f"%{q}%", limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_avg_ratings_for_movies(movie_ids):
    """Map movie_id -> mean ML rating (1–5) for rows that have at least one rating."""
    if not movie_ids:
        return {}
    conn = get_db()
    qs = ",".join("?" * len(movie_ids))
    rows = conn.execute(
        f"SELECT movie_id, AVG(rating) AS ar FROM ratings WHERE movie_id IN ({qs}) "
        "GROUP BY movie_id",
        tuple(int(x) for x in movie_ids),
    ).fetchall()
    conn.close()
    return {int(r["movie_id"]): float(r["ar"]) for r in rows if r["ar"] is not None}


def find_catalog_match_for_external_title(external_title: str, year):
    """
    Map a TMDb (or other) title + year to an ml-1m movie_id using fuzzy title match.
    Returns None if no good match.
    """
    ext = _norm_title(external_title)
    if len(ext) < 2:
        return None
    conn = get_db()
    if year is not None:
        try:
            y = int(year)
        except (TypeError, ValueError):
            y = None
    else:
        y = None
    if y is not None:
        rows = conn.execute(
            "SELECT movie_id, title, year FROM movies "
            "WHERE year IS NOT NULL AND year BETWEEN ? AND ?",
            (y - 1, y + 1),
        ).fetchall()
    else:
        rows = conn.execute("SELECT movie_id, title, year FROM movies",).fetchall()
    conn.close()
    best_id, best_r = None, 0.0
    for r in rows:
        local = _norm_title(r["title"])
        r1 = difflib.SequenceMatcher(None, ext, local).ratio()
        if r1 > best_r:
            best_r, best_id = r1, int(r["movie_id"])
    if best_r >= 0.56 and best_id is not None:
        return best_id
    return None


def get_trending_movies(limit: int = 20):
    """Top movies by average rating with at least 50 reviews."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT m.movie_id, m.title, m.genres, m.year,
               m.tmdb_poster_url, m.tmdb_overview,
               AVG(r.rating) AS avg_rating,
               COUNT(r.rating) AS num_ratings
        FROM movies m
        JOIN ratings r ON m.movie_id = r.movie_id
        GROUP BY m.movie_id
        HAVING num_ratings >= 50
        ORDER BY avg_rating DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_movies(limit: int = 30, min_year: int = None):
    """
    Newest-released rows in the catalog (by `year` descending), for blending into recommendations.
    """
    if min_year is None:
        min_year = Config.LATEST_MIN_YEAR
    conn = get_db()
    rows = conn.execute(
        """
        SELECT movie_id, title, genres, year, tmdb_poster_url, tmdb_overview
        FROM movies
        WHERE year IS NOT NULL AND year >= ?
        ORDER BY year DESC, movie_id DESC
        LIMIT ?
        """,
        (int(min_year), int(limit)),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_rating_count(user_id: int):
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    conn.close()
    return count


# ──────────────────────────────────────────────
# Schema migration + TMDb supplement rows
# ──────────────────────────────────────────────
def ensure_movies_migrated():
    """Add tmdb_id / source to existing DBs (CREATE TABLE is only for new installs)."""
    conn = get_db()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(movies)").fetchall()}
    except sqlite3.OperationalError:
        conn.close()
        return
    if "tmdb_id" not in cols:
        conn.execute("ALTER TABLE movies ADD COLUMN tmdb_id INTEGER")
    if "source" not in cols:
        conn.execute("ALTER TABLE movies ADD COLUMN source TEXT DEFAULT 'ml1m'")
    conn.execute("UPDATE movies SET source = 'ml1m' WHERE source IS NULL")
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_movies_tmdb_id_unique "
            "ON movies(tmdb_id) WHERE tmdb_id IS NOT NULL"
        )
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def is_synthetic_movie_id(movie_id) -> bool:
    try:
        return int(movie_id) >= Config.SYNTHETIC_MOVIE_ID_BASE
    except (TypeError, ValueError):
        return False


def get_next_synthetic_movie_id(conn):
    m = conn.execute(
        "SELECT MAX(movie_id) FROM movies WHERE movie_id >= ?",
        (Config.SYNTHETIC_MOVIE_ID_BASE,),
    ).fetchone()[0]
    if m is None:
        return Config.SYNTHETIC_MOVIE_ID_BASE
    return int(m) + 1


def get_movie_by_tmdb_id(tmdb_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM movies WHERE tmdb_id = ?", (tmdb_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def insert_tmdb_supplement(
    tmdb_id: int,
    title: str,
    year,
    genres_pipe: str,
    poster_url: str,
    overview: str,
):
    """Insert a catalog row for a TMDb-only title; return movie_id. Idempotent on tmdb_id."""
    ex = get_movie_by_tmdb_id(tmdb_id)
    if ex:
        return ex["movie_id"]
    conn = get_db()
    try:
        mid = get_next_synthetic_movie_id(conn)
        conn.execute(
            "INSERT INTO movies (movie_id, title, genres, year, tmdb_poster_url, tmdb_overview, tmdb_id, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'tmdb')",
            (
                mid,
                title,
                genres_pipe or "Drama",
                int(year) if year is not None else None,
                (poster_url or "")[:2000],
                (overview or "")[:8000] if overview else "",
                tmdb_id,
            ),
        )
        conn.commit()
        return mid
    except sqlite3.IntegrityError:
        conn.rollback()
        ex = get_movie_by_tmdb_id(tmdb_id)
        if ex:
            return ex["movie_id"]
        raise
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Standalone: seed the DB
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("🎬 CineMatch — Database Setup")
    conn = get_db()
    create_tables(conn)
    ensure_movies_migrated()

    print("\n📥 Loading ml-1m dataset files…")
    movies_df  = load_movies_df()
    ratings_df = load_ratings_df()
    users_df   = load_users_df()
    print(f"   Movies : {len(movies_df):,}")
    print(f"   Ratings: {len(ratings_df):,}")
    print(f"   Users  : {len(users_df):,}")

    print("\n📀 Seeding database…")
    seed_movies(conn, movies_df)
    seed_ratings(conn, ratings_df)

    conn.close()
    print("\n🎉 Done! Database ready at:", Config.DATABASE_PATH)
