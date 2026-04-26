"""Unit tests: database helper logic (no Flask client)."""
import pytest

from database import _extract_year, get_latest_movies


@pytest.mark.unit
def test_extract_year_from_parentheses():
    assert _extract_year("Toy Story (1995)") == 1995


@pytest.mark.unit
def test_extract_year_no_year():
    assert _extract_year("Mystery Title") is None


@pytest.mark.unit
def test_extract_year_no_trailing_match_without_year_in_parentheses_at_end():
    # Title with something after the closing paren: no match (regex anchors at end of string)
    assert _extract_year("Some Film (2001) The Return") is None


@pytest.mark.unit
def test_extract_year_valid_strict():
    assert _extract_year("Name (2010)") == 2010


@pytest.mark.unit
def test_get_latest_movies_newest_first():
    rows = get_latest_movies(limit=5, min_year=1990)
    if len(rows) < 2:
        return
    y0, y1 = rows[0].get("year"), rows[1].get("year")
    assert y0 is not None and y1 is not None
    assert y0 >= y1
