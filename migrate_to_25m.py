"""
One-time migration: ML-1M → ML-25M.

Steps:
  1. Remove ML-1M movie rows (source='ml1m') from cinematch.db
  2. Seed the 62,423 ML-25M movies (source='ml25m')
  3. Delete stale model artefacts so --build writes fresh ones
  4. Print next steps (model build)
"""
import os
import sys

# Force 25M mode before importing anything from this project
os.environ["MOVIELENS_DATASET"] = "25m"

import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import Config
from database import get_db, load_movies_df, create_tables, ensure_movies_migrated

# ── 1. Safety checks ────────────────────────────────────────────────────────
assert Config.DATASET_USE_25M, "MOVIELENS_DATASET not recognised as 25m"
assert os.path.exists(Config.MOVIES_CSV), f"movies.csv not found at {Config.MOVIES_CSV}"
assert os.path.exists(Config.RATINGS_CSV), f"ratings.csv not found at {Config.RATINGS_CSV}"
print("✅ 25M CSV files found.")

# ── 2. Remove ML-1M movies (keep TMDb supplements >= SYNTHETIC_MOVIE_ID_BASE) ─
conn = get_db()

ml1m_count = conn.execute("SELECT COUNT(*) FROM movies WHERE source = 'ml1m'").fetchone()[0]
print(f"\n🗑  Removing {ml1m_count:,} ML-1M movie rows (source='ml1m')…")
conn.execute("DELETE FROM movies WHERE source = 'ml1m'")
conn.commit()

remaining = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
print(f"   Remaining rows (TMDb supplements): {remaining}")

# ── 3. Seed 25M movies ───────────────────────────────────────────────────────
print(f"\n📥 Loading ml-25m/movies.csv…")
movies_df = load_movies_df()
print(f"   Loaded {len(movies_df):,} movies from CSV.")

rows = []
for r in movies_df.itertuples():
    rows.append((
        int(r.movie_id),
        r.title,
        r.genres,
        int(r.year) if pd.notna(r.year) else None,
        '', '', None, 'ml25m',
    ))

print(f"   Inserting {len(rows):,} rows into movies table…")
conn.executemany(
    "INSERT OR IGNORE INTO movies "
    "(movie_id, title, genres, year, tmdb_poster_url, tmdb_overview, tmdb_id, source) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    rows,
)
conn.commit()

seeded = conn.execute("SELECT COUNT(*) FROM movies WHERE source='ml25m'").fetchone()[0]
print(f"✅ Seeded {seeded:,} ML-25M movies.")

total = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
rating_count = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
print(f"   Total movies in DB: {total:,}  |  Ratings (kept from ML-1M): {rating_count:,}")
conn.close()

# ── 4. Remove stale model artefacts ─────────────────────────────────────────
stale = [
    Config.COSINE_SIM_PATH,
    Config.MOVIE_IDX_PATH,
    Config.TFIDF_PATH,
    Config.COLLAB_MAPPINGS_PATH,
    Config.PREDICTED_PATH,
    Config.USER_ITEM_PATH,
    Config.SVD_MODEL_PATH,
]
print("\n🗑  Removing stale model artefacts…")
for p in stale:
    if os.path.exists(p):
        os.remove(p)
        print(f"   Deleted: {os.path.basename(p)}")

print("\n✅ Migration complete.")
print("\n🔨 Next step — rebuild models (this takes ~10–20 min):")
print("   venv/bin/python recommender.py --build")
