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
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from config import Config
from database import (
    get_avg_ratings_for_movies,
    get_movie_by_id,
    load_movies_df,
    load_ratings_df,
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
        # Favor genre overlap over title text (TMDb non‑Latin titles match poorly by string).
        s = 0.62 * j + 0.22 * t + y_bonus
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
    Build TF-IDF on genres + title keywords (no year) and compute cosine similarity.
    Year is intentionally excluded: it has very high IDF on large catalogs and would
    cluster movies by release year instead of by genre/theme.
    Saves: tfidf_matrix.pkl, cosine_sim.pkl, movie_indices.pkl
    """
    print("\n📐 Building content-based model…")

    def make_features(row):
        # Genres: "Action Crime Thriller" (pipe → space)
        genres = (row['genres'] or '').replace('|', ' ')
        # Title keywords: strip trailing "(YYYY)" then tokenize
        raw_title = re.sub(r'\s*\(\d{4}\)\s*$', '', str(row['title'] or ''))
        # Repeat genres 3× so genre similarity outweighs title word overlap
        return f"{genres} {genres} {genres} {raw_title}"

    movies_df = movies_df.copy()
    movies_df['features'] = movies_df.apply(make_features, axis=1)

    # TF-IDF — no year token, so genre terms dominate similarity scoring
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['features'])
    print(f"   TF-IDF matrix: {tfidf_matrix.shape}")

    n_movies = len(movies_df)
    use_sparse_content = Config.DATASET_USE_25M or n_movies > 12_000
    movie_indices = pd.Series(movies_df.index, index=movies_df['movie_id'])

    _save(tfidf_matrix, Config.TFIDF_PATH)
    _save(movie_indices, Config.MOVIE_IDX_PATH)
    if use_sparse_content:
        print(
            "   Sparse content mode: skipping dense cosine matrix "
            f"(query-time similarity for {n_movies:,} titles)."
        )
        if os.path.exists(Config.COSINE_SIM_PATH):
            try:
                os.remove(Config.COSINE_SIM_PATH)
            except OSError:
                pass
    else:
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print(f"   Cosine sim matrix: {cosine_sim.shape}")
        _save(cosine_sim, Config.COSINE_SIM_PATH)

    print("✅ Content model saved.")
    return None if use_sparse_content else True, movie_indices


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
    tfidf_matrix=None,
):
    """Return Series of {movie_id: content_score} for top_n similar movies."""
    if movie_id not in movie_indices.index:
        return pd.Series(dtype=float)
    idx = int(movie_indices[movie_id])

    if cosine_sim is not None:
        n = int(cosine_sim.shape[0])
        if pos_to_mid is None or len(pos_to_mid) < n:
            pos_to_mid = _content_pos_to_mid_array(movie_indices, n)
        if idx < 0 or idx >= n:
            return pd.Series(dtype=float)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
        movie_ids, scores = [], []
        for row_idx, sc in sim_scores:
            p = int(row_idx)
            if p < 0 or p >= n:
                continue
            mid = int(pos_to_mid[p])
            if mid < 0:
                continue
            movie_ids.append(mid)
            scores.append(sc)
        return pd.Series(scores, index=movie_ids)

    if tfidf_matrix is None:
        return pd.Series(dtype=float)
    n = int(tfidf_matrix.shape[0])
    if pos_to_mid is None or len(pos_to_mid) < n:
        pos_to_mid = _content_pos_to_mid_array(movie_indices, n)
    if idx < 0 or idx >= n:
        return pd.Series(dtype=float)
    sims = linear_kernel(tfidf_matrix[idx : idx + 1], tfidf_matrix).ravel()
    pairs = []
    for j, sc in enumerate(sims):
        if j == idx:
            continue
        mid = int(pos_to_mid[j])
        if mid < 0:
            continue
        pairs.append((mid, float(sc)))
    pairs.sort(key=lambda x: -x[1])
    pairs = pairs[:top_n]
    if not pairs:
        return pd.Series(dtype=float)
    return pd.Series([p[1] for p in pairs], index=[p[0] for p in pairs])


# ──────────────────────────────────────────────
# 2. COLLABORATIVE FILTERING ENGINE (SVD)
# ──────────────────────────────────────────────
def build_collab_model(ratings_df, movies_df):
    """
    TruncatedSVD on a sparse user×movie rating matrix.
    Saves collab_mappings.pkl (factors + seen sets). Does not materialize a dense
    predicted matrix (infeasible at MovieLens 25M scale).
    """
    print("\n🤝 Building collaborative filtering model (SVD, sparse)…")

    u_ids, u_inv = np.unique(ratings_df["user_id"].values, return_inverse=True)
    m_ids, m_inv = np.unique(ratings_df["movie_id"].values, return_inverse=True)
    n_u, n_m = len(u_ids), len(m_ids)
    print(f"   User-item matrix: {n_u:,} × {n_m:,} ({len(ratings_df):,} ratings)…")

    data = ratings_df["rating"].values.astype(np.float32, copy=False)
    user_item_sparse = csr_matrix((data, (u_inv, m_inv)), shape=(n_u, n_m))

    svd = TruncatedSVD(
        n_components=50, random_state=42, algorithm="randomized"
    )
    U = svd.fit_transform(user_item_sparse)
    print(f"   Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.3f}")

    print("   Evaluating RMSE on 20% holdout sample…")
    sample = ratings_df.sample(n=min(50_000, len(ratings_df)), random_state=42)
    _, test = train_test_split(sample, test_size=0.2, random_state=42)
    user_row = pd.Series(np.arange(n_u, dtype=np.int32), index=u_ids)
    movie_col = pd.Series(np.arange(n_m, dtype=np.int32), index=m_ids)
    y_true, y_pred = [], []
    for row in test.itertuples(index=False):
        ui = user_row.get(row.user_id)
        mj = movie_col.get(row.movie_id)
        if ui is None or mj is None:
            continue
        pred = float(U[int(ui)].dot(svd.components_[:, int(mj)]))
        y_true.append(float(row.rating))
        y_pred.append(pred)
    if y_true:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        print(f"   ✅ RMSE on test split: {rmse:.4f} (target < 1.8)")

    print("   Building per-user seen sets…")
    seen_by_user = {
        int(uid): set(grp["movie_id"].astype(int).values)
        for uid, grp in ratings_df.groupby("user_id", sort=False)
    }

    bundle = {
        "svd": svd,
        "U": U.astype(np.float32, copy=False),
        "user_ids": u_ids.astype(np.int32, copy=False),
        "movie_ids": m_ids.astype(np.int32, copy=False),
        "seen_by_user": seen_by_user,
    }
    _save(bundle, Config.COLLAB_MAPPINGS_PATH)

    for p in (Config.PREDICTED_PATH, Config.USER_ITEM_PATH, Config.SVD_MODEL_PATH):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    print("✅ Collaborative model saved (collab_mappings.pkl).")
    return bundle


def get_collab_scores(
    user_id,
    predicted_df,
    user_item,
    top_n=50,
    collab_bundle=None,
):
    """Return Series of {movie_id: normalized score} for unseen movies."""
    if collab_bundle is not None:
        seen = collab_bundle["seen_by_user"].get(int(user_id))
        if seen is None:
            return pd.Series(dtype=float)
        u_arr = collab_bundle["user_ids"]
        hits = np.where(u_arr == int(user_id))[0]
        if len(hits) == 0:
            return pd.Series(dtype=float)
        row_idx = int(hits[0])
        Urow = collab_bundle["U"][row_idx]
        pred = Urow @ collab_bundle["svd"].components_
        movie_ids = collab_bundle["movie_ids"]
        s = pd.Series(pred, index=movie_ids.astype(int))
        s = s.drop(index=list(seen), errors="ignore")
        if s.empty:
            return pd.Series(dtype=float)
        preds_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return preds_norm.nlargest(top_n)

    if predicted_df is None or user_item is None:
        return pd.Series(dtype=float)
    if user_id not in predicted_df.index:
        return pd.Series(dtype=float)
    seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
    preds = predicted_df.loc[user_id].drop(index=seen, errors="ignore")
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


def _movie_genre_set(genres) -> set:
    if not genres:
        return set()
    out = set()
    for g in genres:
        if g is None:
            continue
        s = str(g).strip()
        if s:
            out.add(s)
    return out


def _genres_pipe_to_set(genres) -> set:
    """DB/ml-1m pipe string 'A|B' or iterable → genre set."""
    if not genres:
        return set()
    if isinstance(genres, str):
        return {x.strip() for x in genres.split("|") if x.strip()}
    return _movie_genre_set(genres)


def _max_genre_jaccard_overlap(genres_set: set, picked_sets: list) -> float:
    if not picked_sets or not genres_set:
        return 0.0
    best = 0.0
    for ps in picked_sets:
        u = genres_set | ps
        inter = genres_set & ps
        j = len(inter) / max(len(u), 1)
        if j > best:
            best = j
    return best


def diversify_ranked_results(rows: list, top_n: int, genre_penalty: float = 0.14) -> list:
    """
    Greedy re-ranking on an already score-sorted list: keep strong matches but
    penalize titles whose genres closely overlap films already chosen, so rows
    feel less repetitive than pure top-N by hybrid_score alone.
    """
    if not rows or top_n <= 0:
        return []
    pool = list(rows)
    picked = []
    picked_ids = set()
    picked_sets = []

    while pool and len(picked) < top_n:
        best_i = -1
        best_adj = -1e9
        for i, r in enumerate(pool):
            gset = _movie_genre_set(r.get("genres"))
            overlap = _max_genre_jaccard_overlap(gset, picked_sets)
            adj = float(r.get("hybrid_score", 0)) - genre_penalty * overlap
            if adj > best_adj:
                best_adj = adj
                best_i = i
        if best_i < 0:
            break
        take = pool.pop(best_i)
        picked.append(take)
        picked_ids.add(int(take["movie_id"]))
        picked_sets.append(_movie_genre_set(take.get("genres")))

    if len(picked) < top_n:
        for r in rows:
            if len(picked) >= top_n:
                break
            mid = int(r["movie_id"])
            if mid in picked_ids:
                continue
            picked.append(r)
            picked_ids.add(mid)

    return picked[:top_n]


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
        self.tfidf_matrix = None
        self.movie_indices = None
        self.predicted_df  = None
        self.user_item     = None
        self._collab_bundle = None
        self.movies_df     = None
        self._content_pos_to_mid: Optional[np.ndarray] = None
        self._loaded       = False

    def load(self):
        """Load all .pkl models into memory (call once at startup)."""
        if self._loaded:
            return
        print("🔄 Loading recommendation models…")
        self.movie_indices = _load(Config.MOVIE_IDX_PATH)
        self.tfidf_matrix = _load(Config.TFIDF_PATH)
        self.cosine_sim = (
            _load(Config.COSINE_SIM_PATH)
            if os.path.exists(Config.COSINE_SIM_PATH)
            else None
        )
        if self.cosine_sim is None:
            print("   Content: query-time cosine via TF-IDF (large catalog).")
        if os.path.exists(Config.COLLAB_MAPPINGS_PATH):
            self._collab_bundle = _load(Config.COLLAB_MAPPINGS_PATH)
            self.predicted_df = None
            self.user_item = None
        else:
            self._collab_bundle = None
            self.predicted_df = (
                _load(Config.PREDICTED_PATH)
                if os.path.exists(Config.PREDICTED_PATH)
                else None
            )
            self.user_item = (
                _load(Config.USER_ITEM_PATH)
                if os.path.exists(Config.USER_ITEM_PATH)
                else None
            )
        self.movies_df = load_movies_df()
        if self.cosine_sim is not None:
            n = int(self.cosine_sim.shape[0])
        else:
            n = int(self.tfidf_matrix.shape[0])
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
        anchor_row = get_movie_by_id(orig_id)
        if eff_id not in self.movie_indices.index:
            if not anchor_row:
                from database import get_trending_movies
                return get_trending_movies(top_n), {
                    "fallback": "unknown_anchor", "user_anchor_id": orig_id,
                }
            eff_id, recommend_meta = nearest_ml1m_proxy_mid(anchor_row, self.movies_df)
            if recommend_meta:
                recommend_meta["recommendation_mode"] = "tmdb_proxy"

        anchor_g = _genres_pipe_to_set((anchor_row or {}).get("genres"))
        profile_for_persona = dict(user_profile)
        if recommend_meta and recommend_meta.get("used_proxy") and anchor_g:
            merged = set(profile_for_persona.get("preferred_genres") or []) | anchor_g
            profile_for_persona["preferred_genres"] = list(merged)

        # 1. Content similarity (MovieLens movie_id index)
        content_scores = get_content_scores(
            eff_id,
            self.cosine_sim,
            self.movie_indices,
            self._content_pos_to_mid,
            top_n=100,
            tfidf_matrix=self.tfidf_matrix,
        )

        # Determine cold-start
        from database import get_user_rating_count
        rating_count = get_user_rating_count(user_id)
        is_cold_start = rating_count < Config.COLD_START_THRESHOLD

        used_proxy = bool(recommend_meta and recommend_meta.get("used_proxy"))
        if used_proxy:
            # Proxy similarity is noisy for non‑Western anchors; lean on taste + anchor genres.
            if is_cold_start:
                w_c, w_collab, w_p = 0.52, 0.0, 0.48
            else:
                w_c, w_collab, w_p = 0.42, 0.18, 0.40
        elif is_cold_start:
            w_c, w_collab, w_p = 0.8, 0.0, 0.2
        else:
            w_c, w_collab, w_p = 0.5, 0.3, 0.2

        collab_scores = (
            get_collab_scores(
                user_id,
                self.predicted_df,
                self.user_item,
                top_n=100,
                collab_bundle=self._collab_bundle,
            )
            if not is_cold_start
            else pd.Series(dtype=float)
        )

        persona_scores = get_personalization_scores(profile_for_persona, self.movies_df)

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

        def collect_scored_rows(apply_era_filter: bool):
            """apply_era_filter=False skips era gate (used when ML catalog years never match UI eras)."""
            rows_out = []
            for mid in candidates:
                c_score = float(content_scores.get(mid, 0))
                co_score = float(collab_scores.get(mid, 0))
                p_score = float(persona_scores.get(mid, 0))
                hybrid = w_c * c_score + w_collab * co_score + w_p * p_score

                if min_avg_rating is not None and min_avg_rating > 0:
                    mean_rating = avgs.get(int(mid))
                    if mean_rating is not None and mean_rating < min_avg_rating:
                        continue

                if era_filter and apply_era_filter:
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

                row = self.movies_df[self.movies_df['movie_id'] == mid]
                if row.empty:
                    continue
                row = row.iloc[0]

                cand_g = _genres_pipe_to_set(row.get("genres"))
                if anchor_g:
                    u_ag = anchor_g | cand_g
                    hybrid += 0.14 * (len(anchor_g & cand_g) / max(len(u_ag), 1))

                hybrid = min(hybrid, 1.25)
                rows_out.append({
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
            return rows_out

        results = collect_scored_rows(True)
        # Era filter is always respected — if nothing matched (e.g. "Modern" on ML-1M which
        # only has movies up to ~2000), results stays empty and the app-layer blends
        # (_merge_tmdb_popular / _merge_regional) fill those slots with modern content.

        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        skip_ids = {int(orig_id), int(eff_id)}
        results = [r for r in results if int(r['movie_id']) not in skip_ids]
        results = diversify_ranked_results(results, top_n)
        return results, recommend_meta


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
