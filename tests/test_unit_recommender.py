"""Unit tests: recommender constants and light structure (no .pkl load)."""
import numpy as np
import pandas as pd
import pytest

from recommender import (
    ALL_GENRES,
    AGE_ERA_MAP,
    ERA_YEAR_RANGES,
    _content_pos_to_mid_array,
    _genres_pipe_to_set,
    diversify_ranked_results,
    get_content_scores,
)


@pytest.mark.unit
def test_era_year_ranges_cover_decades():
    assert ERA_YEAR_RANGES["90s"] == (1990, 1999)
    assert ERA_YEAR_RANGES["modern"][0] == 2010


@pytest.mark.unit
def test_age_era_map_keys_are_ml1m_age_codes():
    assert 1 in AGE_ERA_MAP
    assert 18 in AGE_ERA_MAP
    assert 56 in AGE_ERA_MAP


@pytest.mark.unit
def test_all_genres_count():
    assert len(ALL_GENRES) == 18


@pytest.mark.unit
def test_genres_pipe_to_set_splits_ml_format():
    assert _genres_pipe_to_set("Comedy|Romance|Drama") == {"Comedy", "Romance", "Drama"}
    assert _genres_pipe_to_set("") == set()


@pytest.mark.unit
def test_get_content_scores_maps_rows_via_pos_to_mid_not_iloc():
    """
    Content similarity indices refer to the matrix built at model-train time; they
    must not be applied as `movies_df.iloc[...]` when the current CSV is shorter
    (or reordered) than the saved cosine matrix.
    """
    # 3x3: anchor mid=10 at position 0; other mids 20, 30 at 1, 2
    cos = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.0],
            [0.1, 0.0, 1.0],
        ],
        dtype=float,
    )
    movie_indices = pd.Series([0, 1, 2], index=[10, 20, 30])
    pos = _content_pos_to_mid_array(movie_indices, 3)
    s = get_content_scores(10, cos, movie_indices, pos, top_n=2)
    assert s.index.tolist() == [20, 30]
    assert s.iloc[0] == pytest.approx(0.2)


@pytest.mark.unit
def test_diversify_ranked_results_prefers_genre_variety():
    rows = [
        {"movie_id": 1, "hybrid_score": 1.0, "genres": ["Action", "Thriller"]},
        {"movie_id": 2, "hybrid_score": 0.95, "genres": ["Action", "Thriller"]},
        {"movie_id": 3, "hybrid_score": 0.9, "genres": ["Romance", "Drama"]},
    ]
    out = diversify_ranked_results(rows, top_n=2, genre_penalty=0.5)
    ids = [r["movie_id"] for r in out]
    assert ids[0] == 1
    assert 3 in ids
    assert len(out) == 2
