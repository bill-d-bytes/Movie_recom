"""
recommender.py — ML engine for CineMatch.

Three engines:
  1. Content-Based   (TF-IDF + cosine similarity on genres/title/year)
  2. Collaborative   (Truncated SVD on user-item rating matrix)
  3. Hybrid          (weighted combination + personalization)

Usage:
  # Build + save models (run once):
  python recommender.py --build

  # Then load at app startup:
  engine = RecommendationEngine()
  engine.load()
  results = engine.hybrid_recommend(user_id=1, movie_id=1, user_profile={...})
"""
import os
import re
import difflib
import pickle
import argparse
from typing import Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from config import Config
from database import (
    get_avg_ratings_for_movies,
    get_movie_by_id,
    load_movies_df,
    load_ratings_df,
    load_users_df,
)


# ──────────────────────────────────────────────
# Age-code → era mapping  (ml-1m specific)
# ──────────────────────────────────────────────
# Maps ml-1m age codes to preferred decade bias
AGE_ERA_MAP = {
    1:  ['modern', '00s'],          # Under 18
    18: ['modern', '00s'],          # 18-24
    25: ['00s', '90s'],             # 25-34
    35: ['90s', '80s'],             # 35-44
    45: ['80s', '90s'],             # 45-49
    50: ['80s', 'classic'],         # 50-55
    56: ['classic', '80s'],         # 56+
}

ERA_YEAR_RANGES = {
    'classic': (1900, 1979),
    '80s':     (1980, 1989),
    '90s':     (1990, 1999),
    '00s':     (2000, 2009),
    'modern':  (2010, 2030),
}


def _norm_anchor_title(s: str) -> str:
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", (s or "")).strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
    return " ".join(t.split())


def nearest_ml1m_proxy_mid(anchor_row, movies_df, synthetic_base=None):
    """
    Pick an ml-1m movie_id (below synthetic_base) to stand in for a TMDb-only anchor
    so TF-IDF / SVD indices apply. Returns (proxy_id, meta dict).
    """
    base = synthetic_base if synthetic_base is not None else Config.SYNTHETIC_MOVIE_ID_BASE
    g_user = {x.strip() for x in (anchor_row.get("genres") or "").split("|") if x.strip()}
    title_u = _norm_anchor_title(anchor_row.get("title", ""))
    y_u = anchor_row.get("year")
    if y_u is not None and pd.isna(y_u):
        y_u = None
    if y_u is not None:
        try:
            y_u = int(y_u)
        except (TypeError, ValueError):
            y_u = None

    best_mid, best_s = 1, 0.0
    for _, m in movies_df.iterrows():
        mid = int(m["movie_id"])
        if mid >= base:
            continue
        g = {x.strip() for x in m["genres"].split("|")} if m.get("genres") else set()
        union = g_user | g
        j = (len(g_user & g) / (len(union) + 1e-9)) if union else 0.0
        t = difflib.SequenceMatcher(
            None, title_u, _norm_anchor_title(str(m.get("title", "")))
        ).ratio()
        y_bonus = 0.0
        my = m.get("year")
        if y_u is not None and my is not None and not (isinstance(my, float) and my != my):
            try:
                y_bonus = 0.15 * max(0.0, 1.0 - min(8, abs(int(y_u) - int(my))) / 8.0)
            except (TypeError, ValueError):
                pass
        s = 0.55 * j + 0.32 * t + y_bonus
        if s > best_s:
            best_s, best_mid = s, mid

    ptitle = ""
    rowp = movies_df[movies_df["movie_id"] == best_mid]
    if not rowp.empty:
        ptitle = str(rowp.iloc[0]["title"])

    if best_s < 0.1 and 1 in set(movies_df["movie_id"]):
        best_mid = 1
        row1 = movies_df[movies_df["movie_id"] == 1]
        if not row1.empty:
            ptitle = str(row1.iloc[0].get("title", ""))

    meta = {
        "used_proxy": True,
        "proxy_movie_id": int(best_mid),
        "proxy_title": ptitle,
        "user_anchor_id": int(anchor_row.get("movie_id", 0) or 0),
        "user_anchor_title": str(anchor_row.get("title", "") or ""),
        "match_strength": round(float(best_s), 3),
    }
    return int(best_mid), meta

# ml-1m genre list (all 18 genres)
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]


def _save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"   💾 Saved → {os.path.basename(path)}")


def _load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# 1. CONTENT-BASED ENGINE
# ──────────────────────────────────────────────
def build_content_model(movies_df: pd.DataFrame):
    """
    Build TF-IDF on 'genres + year' and compute cosine similarity.
    Saves: tfidf_matrix.pkl, cosine_sim.pkl, movie_indices.pkl
    """
    print("\n📐 Building content-based model…")

    # Feature string: replace pipe with space, append year
    def make_features(row):
        genres = row['genres'].replace('|', ' ')
        year   = str(row['year']) if pd.notna(row['year']) else ''
        return f"{genres} {year}"

    movies_df = movies_df.copy()
    movies_df['features'] = movies_df.apply(make_features, axis=1)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['features'])
    print(f"   TF-IDF matrix: {tfidf_matrix.shape}")

    # Compute cosine similarity (dense — ~3900×3900 ≈ 120 MB, acceptable)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"   Cosine sim matrix: {cosine_sim.shape}")

    # movie_id → index mapping
    movie_indices = pd.Series(movies_df.index, index=movies_df['movie_id'])

    _save(tfidf_matrix, Config.TFIDF_PATH)
    _save(cosine_sim,   Config.COSINE_SIM_PATH)
    _save(movie_indices, Config.MOVIE_IDX_PATH)
    print("✅ Content model saved.")
    return cosine_sim, movie_indices


def _content_pos_to_mid_array(movie_indices: pd.Series, n: int) -> np.ndarray:
    """
    Map matrix row/column index → movie_id (as stored when the content model was built).
    Cosine columns always align with build-time row order, which may be longer than
    the current `load_movies_df()` CSV, so we must not use `movies_df.iloc[...]`.
    """
    out = np.full(n, -1, dtype=np.int64)
    for mid, pos in movie_indices.items():
        p = int(pos)
        if 0 <= p < n:
            out[p] = int(mid)
    return out


def get_content_scores(
    movie_id,
    cosine_sim,
    movie_indices,
    pos_to_mid: np.ndarray,
    top_n=50,
):
    """Return Series of {movie_id: content_score} for top_n similar movies."""
    if movie_id not in movie_indices.index:
        return pd.Series(dtype=float)
    n = int(cosine_sim.shape[0])
    if pos_to_mid is None or len(pos_to_mid) < n:
        pos_to_mid = _content_pos_to_mid_array(movie_indices, n)
    idx = int(movie_indices[movie_id])
    if idx < 0 or idx >= n:
        return pd.Series(dtype=float)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_ids, scores = [], []
    for row_idx, sc in sim_scores:
        p = int(row_idx)
        if 0 > p or p >= n:
            continue
        mid = int(pos_to_mid[p])
        if mid < 0:
            continue
        movie_ids.append(mid)
        scores.append(sc)
    return pd.Series(scores, index=movie_ids)


# ──────────────────────────────────────────────
# 2. COLLABORATIVE FILTERING ENGINE (SVD)
# ──────────────────────────────────────────────
def build_collab_model(ratings_df, movies_df):
    """
    Build Truncated SVD on the user-item rating matrix.
    Saves: svd_model.pkl, user_item_matrix.pkl, predicted_ratings.pkl
    Also prints RMSE on 20% test split.
    """
    print("\n🤝 Building collaborative filtering model (SVD)…")

    # Build user-item pivot (sparse)
    print("   Building user-item matrix (6040 × 3883)…")
    user_item = ratings_df.pivot_table(
        index='user_id', columns='movie_id', values='rating', fill_value=0
    )
    user_item_sparse = csr_matrix(user_item.values)

    # SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd.fit(user_item_sparse)
    predicted = svd.inverse_transform(svd.transform(user_item_sparse))
    predicted_df = pd.DataFrame(predicted,
                                index=user_item.index,
                                columns=user_item.columns)
    print(f"   Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.3f}")

    # RMSE evaluation on a sample
    print("   Evaluating RMSE on 20% holdout sample…")
    sample = ratings_df.sample(n=min(50_000, len(ratings_df)), random_state=42)
    train, test = train_test_split(sample, test_size=0.2, random_state=42)
    y_true, y_pred = [], []
    for row in test.itertuples():
        uid, mid = row.user_id, row.movie_id
        if uid in predicted_df.index and mid in predicted_df.columns:
            y_true.append(row.rating)
            y_pred.append(predicted_df.loc[uid, mid])
    if y_true:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        print(f"   ✅ RMSE on test split: {rmse:.4f} (target < 1.8)")

    _save(svd,          Config.SVD_MODEL_PATH)
    _save(user_item,    Config.USER_ITEM_PATH)
    _save(predicted_df, Config.PREDICTED_PATH)
    print("✅ Collaborative model saved.")
    return predicted_df, user_item


def get_collab_scores(user_id, predicted_df,
                      user_item, top_n=50):
    """Return Series of {movie_id: predicted_rating} for unseen movies."""
    if user_id not in predicted_df.index:
        return pd.Series(dtype=float)
    seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
    preds = predicted_df.loc[user_id].drop(index=seen, errors='ignore')
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
    return preds_norm.nlargest(top_n)


# ──────────────────────────────────────────────
# 3. PERSONALIZATION SCORING
# ──────────────────────────────────────────────
def get_personalization_scores(user_profile, movies_df):
    """
    Score every movie based on:
      - Genre overlap with user's preferred_genres (Jaccard)
      - Age-based era preference (maps ml-1m age codes)
      - Gender affinity (lightweight heuristic from genre stats)
    Returns a normalised Series {movie_id: score}.
    """
    preferred_genres = set(user_profile.get('preferred_genres', []))
    age_code         = user_profile.get('age', 25)
    gender           = user_profile.get('gender', 'M')

    # Map age code to nearest ml-1m bucket
    age_buckets = [1, 18, 25, 35, 45, 50, 56]
    nearest_age = min(age_buckets, key=lambda x: abs(x - age_code))
    preferred_eras = AGE_ERA_MAP.get(nearest_age, ['modern'])

    scores = {}
    for row in movies_df.itertuples():
        movie_genres = set(row.genres.split('|'))
        year = row.year if pd.notna(row.year) else 2000

        # 1. Genre Jaccard similarity
        if preferred_genres:
            intersection = len(preferred_genres & movie_genres)
            union = len(preferred_genres | movie_genres)
            genre_score = intersection / union if union > 0 else 0
        else:
            genre_score = 0.5   # neutral if no preference set

        # 2. Era score
        era_score = 0.0
        for era in preferred_eras:
            start, end = ERA_YEAR_RANGES[era]
            if start <= year <= end:
                era_score = 1.0
                break
        # Partial credit for adjacent eras
        if era_score == 0:
            era_score = 0.2

        # 3. Gender affinity (simple heuristic based on genre patterns)
        gender_score = 1.0
        if gender == 'F':
            if movie_genres & {'Romance', 'Drama', 'Musical'}:
                gender_score = 1.2
            elif movie_genres & {'Horror', 'War', 'Western'}:
                gender_score = 0.8
        elif gender == 'M':
            if movie_genres & {'Action', 'Sci-Fi', 'Thriller', 'Crime'}:
                gender_score = 1.2
            elif movie_genres & {'Romance', 'Musical'}:
                gender_score = 0.8

        # Combine (clamp gender to 1.0 max for final normalisation)
        raw = genre_score * 0.6 + era_score * 0.4
        raw *= min(gender_score, 1.2)
        scores[row.movie_id] = raw

    series = pd.Series(scores)
    # Normalise to [0, 1]
    if series.max() > 0:
        series = (series - series.min()) / (series.max() - series.min() + 1e-9)
    return series


# ──────────────────────────────────────────────
# 4. HYBRID ENGINE
# ──────────────────────────────────────────────
class RecommendationEngine:
    """
    Load pre-built models and expose hybrid_recommend().
    Call engine.load() once at Flask startup.
    """

    def __init__(self):
        self.cosine_sim   = None
        self.movie_indices = None
        self.predicted_df  = None
        self.user_item     = None
        self.movies_df     = None
        self._content_pos_to_mid: Optional[np.ndarray] = None
        self._loaded       = False

    def load(self):
        """Load all .pkl models into memory (call once at startup)."""
        if self._loaded:
            return
        print("🔄 Loading recommendation models…")
        self.cosine_sim    = _load(Config.COSINE_SIM_PATH)
        self.movie_indices = _load(Config.MOVIE_IDX_PATH)
        self.predicted_df  = _load(Config.PREDICTED_PATH)
        self.user_item     = _load(Config.USER_ITEM_PATH)
        self.movies_df     = load_movies_df()
        n = int(self.cosine_sim.shape[0])
        self._content_pos_to_mid = _content_pos_to_mid_array(self.movie_indices, n)
        self._loaded = True
        print("✅ Models loaded.")

    def hybrid_recommend(self, user_id: int, movie_id: int,
                         user_profile: dict, top_n: int = None,
                         era_filter: list = None, min_avg_rating: float = None):
        """
        Returns (list of top-N movie dicts, recommend_meta or None).
        TMDb-only / synthetic movie_ids map to a nearest ml-1m id for the ML models; meta explains the proxy.
        """
        if not self._loaded:
            self.load()

        top_n = top_n or Config.TOP_N_DEFAULT
        top_n = min(top_n, Config.TOP_N_MAX)

        recommend_meta = None
        orig_id = int(movie_id)
        eff_id = orig_id
        if eff_id not in self.movie_indices.index:
            ar = get_movie_by_id(eff_id)
            if not ar:
                from database import get_trending_movies
                return get_trending_movies(top_n), {
                    "fallback": "unknown_anchor", "user_anchor_id": orig_id,
                }
            eff_id, recommend_meta = nearest_ml1m_proxy_mid(ar, self.movies_df)
            if recommend_meta:
                recommend_meta["recommendation_mode"] = "tmdb_proxy"

        # 1. Get individual scores (ml-1m id only)
        content_scores = get_content_scores(
            eff_id, self.cosine_sim, self.movie_indices,
            self._content_pos_to_mid, top_n=100
        )

        # Determine cold-start
        from database import get_user_rating_count
        rating_count = get_user_rating_count(user_id)
        is_cold_start = rating_count < Config.COLD_START_THRESHOLD

        if is_cold_start:
            w_c, w_collab, w_p = 0.8, 0.0, 0.2
        else:
            w_c, w_collab, w_p = 0.5, 0.3, 0.2

        collab_scores = get_collab_scores(
            user_id, self.predicted_df, self.user_item, top_n=100
        ) if not is_cold_start else pd.Series(dtype=float)

        persona_scores = get_personalization_scores(user_profile, self.movies_df)

        # 2. Build candidate pool (union of content + collab top-100)
        candidates = set(content_scores.index) | set(collab_scores.index)
        if len(candidates) == 0:
            from database import get_trending_movies
            t = get_trending_movies(top_n)
            m = dict(recommend_meta) if recommend_meta else {}
            m["fallback"] = "no_candidates"
            return t, m

        avgs = {}
        if min_avg_rating is not None and min_avg_rating > 0:
            avgs = get_avg_ratings_for_movies(candidates)

        results = []
        for mid in candidates:
            c_score  = float(content_scores.get(mid, 0))
            co_score = float(collab_scores.get(mid, 0))
            p_score  = float(persona_scores.get(mid, 0))
            hybrid   = w_c * c_score + w_collab * co_score + w_p * p_score

            if min_avg_rating is not None and min_avg_rating > 0:
                ar = avgs.get(int(mid))
                if ar is not None and ar < min_avg_rating:
                    continue

            # Era filter (applied post-scoring)
            if era_filter:
                movie_row = self.movies_df[self.movies_df['movie_id'] == mid]
                if not movie_row.empty:
                    year = movie_row.iloc[0]['year']
                    if pd.notna(year):
                        in_era = any(
                            ERA_YEAR_RANGES[e][0] <= year <= ERA_YEAR_RANGES[e][1]
                            for e in era_filter if e in ERA_YEAR_RANGES
                        )
                        if not in_era:
                            continue

            # Get movie info
            row = self.movies_df[self.movies_df['movie_id'] == mid]
            if row.empty:
                continue
            row = row.iloc[0]

            results.append({
                'movie_id':       int(mid),
                'title':          row['title'],
                'genres':         row['genres'].split('|'),
                'year':           int(row['year']) if pd.notna(row['year']) else None,
                'hybrid_score':   round(hybrid, 4),
                'content_score':  round(c_score, 4),
                'collab_score':   round(co_score, 4),
                'persona_score':  round(p_score, 4),
                'is_cold_start':  is_cold_start,
            })

        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:top_n], recommend_meta


# ──────────────────────────────────────────────
# CLI: build models
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true',
                        help='Build and save all recommendation models')
    args = parser.parse_args()

    if args.build:
        print("🎬 CineMatch — Building ML Models")
        print("📥 Loading dataset…")
        movies_df  = load_movies_df()
        ratings_df = load_ratings_df()
        print(f"   {len(movies_df):,} movies, {len(ratings_df):,} ratings")

        build_content_model(movies_df)
        build_collab_model(ratings_df, movies_df)
        print("\n🎉 All models built and saved to models/")
    else:
        parser.print_help()
