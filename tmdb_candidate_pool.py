from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


TMDB_CANDIDATE_POOL_CSV = "artifacts/tmdb/tmdb_genre_candidates.csv"

MOVIELENS_TO_TMDB_GENRES: Dict[str, List[str]] = {
    "Action": ["Action"],
    "Adventure": ["Adventure"],
    "Animation": ["Animation"],
    "Children's": ["Family"],
    "Comedy": ["Comedy"],
    "Crime": ["Crime"],
    "Documentary": ["Documentary"],
    "Drama": ["Drama"],
    "Fantasy": ["Fantasy"],
    "Film-Noir": ["Crime", "Mystery"],
    "Horror": ["Horror"],
    "Musical": ["Music"],
    "Mystery": ["Mystery"],
    "Romance": ["Romance"],
    "Sci-Fi": ["Science Fiction"],
    "Thriller": ["Thriller"],
    "War": ["War"],
    "Western": ["Western"],
}


def candidate_pool_file_mtime(path: str = TMDB_CANDIDATE_POOL_CSV) -> float:
    pool_path = Path(path)
    return pool_path.stat().st_mtime if pool_path.exists() else 0.0


def _string_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _int_value(value: Any) -> int:
    text = _string_value(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _float_value(value: Any) -> float:
    text = _string_value(value)
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _release_date_value(value: Any) -> int:
    text = _string_value(value)
    if not text:
        return 0
    try:
        return date.fromisoformat(text).toordinal()
    except ValueError:
        return 0


def _merge_pipe_values(left: Any, right: Any) -> str:
    values: List[str] = []
    seen: set[str] = set()
    for raw in (left, right):
        text = _string_value(raw)
        if not text:
            continue
        for item in text.split("|"):
            clean = item.strip()
            if clean and clean not in seen:
                seen.add(clean)
                values.append(clean)
    return "|".join(values)


def candidate_rank_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -_int_value(row.get("release_year", 0)),
        -_release_date_value(row.get("release_date", "")),
        -_float_value(row.get("popularity", 0.0)),
        -_float_value(row.get("vote_average", 0.0)),
        -_int_value(row.get("vote_count", 0)),
        _string_value(row.get("raw_title") or row.get("title", "")).casefold(),
    )


def merge_and_rank_candidates(
    rows: Iterable[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    deduped: Dict[int, Dict[str, Any]] = {}
    fallback_keys: set[str] = set()

    for row in rows:
        normalized = dict(row)
        tmdb_id = _int_value(normalized.get("tmdb_id", 0))
        fallback_key = (
            f"{_string_value(normalized.get('title', '')).casefold()}|"
            f"{_int_value(normalized.get('release_year', 0))}"
        )
        if tmdb_id:
            existing = deduped.get(tmdb_id)
            if existing is None:
                deduped[tmdb_id] = normalized
                continue
            merged = dict(existing)
            merged["source_tmdb_genres"] = _merge_pipe_values(
                existing.get("source_tmdb_genres", ""),
                normalized.get("source_tmdb_genres", ""),
            )
            merged["source_sort_orders"] = _merge_pipe_values(
                existing.get("source_sort_orders", ""),
                normalized.get("source_sort_orders", ""),
            )
            if candidate_rank_sort_key(normalized) < candidate_rank_sort_key(existing):
                merged.update(normalized)
                merged["source_tmdb_genres"] = _merge_pipe_values(
                    existing.get("source_tmdb_genres", ""),
                    normalized.get("source_tmdb_genres", ""),
                )
                merged["source_sort_orders"] = _merge_pipe_values(
                    existing.get("source_sort_orders", ""),
                    normalized.get("source_sort_orders", ""),
                )
            deduped[tmdb_id] = merged
            continue
        if fallback_key in fallback_keys:
            continue
        fallback_keys.add(fallback_key)
        deduped[-len(fallback_keys)] = normalized

    ranked = sorted(deduped.values(), key=candidate_rank_sort_key)
    normalized_rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(ranked[:limit], start=1):
        item = dict(row)
        item["rank_within_genre"] = rank
        normalized_rows.append(item)
    return normalized_rows


def _normalize_candidate_row(row: pd.Series) -> Dict[str, Any]:
    release_year = _int_value(row.get("release_year", ""))
    title = _string_value(row.get("title", ""))
    display_title = title if not release_year else f"{title} ({release_year})"
    return {
        "tmdb_id": _int_value(row.get("tmdb_id", "")),
        "title": display_title,
        "raw_title": title,
        "release_year": release_year,
        "release_date": _string_value(row.get("release_date", "")),
        "popularity": _float_value(row.get("popularity", "")),
        "vote_average": _float_value(row.get("vote_average", "")),
        "vote_count": _int_value(row.get("vote_count", "")),
        "poster_url": _string_value(row.get("poster_url", "")),
        "overview": _string_value(row.get("overview", "")),
        "rank_within_genre": max(1, _int_value(row.get("rank_within_genre", ""))),
    }


def load_tmdb_candidate_pool(
    path: str = TMDB_CANDIDATE_POOL_CSV,
    modified_time: float | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    del modified_time
    pool_path = Path(path)
    if not pool_path.exists():
        raise FileNotFoundError(
            f"TMDb candidate pool not found: {pool_path}. "
            "Run build_tmdb_candidate_pool.py first."
        )

    df = pd.read_csv(pool_path)
    required_columns = {"genre", "title", "release_year", "poster_url", "rank_within_genre"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"TMDb candidate pool is missing required columns: {missing_text}")

    pool: Dict[str, List[Dict[str, Any]]] = {}
    for genre, group in df.groupby("genre", sort=False):
        normalized_rows = [_normalize_candidate_row(row) for _, row in group.iterrows()]
        normalized_rows.sort(
            key=lambda row: (
                row["rank_within_genre"],
                *candidate_rank_sort_key(row),
            )
        )
        pool[str(genre)] = normalized_rows
    return pool


def attach_tmdb_candidates(
    base_result: Dict[str, Any],
    candidate_pool: Dict[str, List[Dict[str, Any]]],
    *,
    top_n_movies_per_genre: int,
) -> Dict[str, Any]:
    enriched_result = {
        key: value
        for key, value in base_result.items()
        if key != "genre_top_movies"
    }

    genre_top_movies: Dict[str, List[Dict[str, Any]]] = {}
    for item in base_result.get("top_genres", []):
        genre = str(item.get("genre", ""))
        genre_score = float(item.get("score", 0.0))
        candidates = candidate_pool.get(genre, [])[:top_n_movies_per_genre]
        rows: List[Dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            row = dict(candidate)
            row["score"] = max(0.0, genre_score - ((rank - 1) * 0.01))
            rows.append(row)
        genre_top_movies[genre] = rows

    enriched_result["genre_top_movies"] = genre_top_movies
    return enriched_result
