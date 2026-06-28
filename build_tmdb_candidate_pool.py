from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from tmdb_candidate_pool import (
    MOVIELENS_TO_TMDB_GENRES,
    TMDB_CANDIDATE_POOL_CSV,
    merge_and_rank_candidates,
)


TMDB_API_ROOT = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
DISCOVER_SORT_ORDERS = (
    "primary_release_date.desc",
    "popularity.desc",
    "vote_average.desc",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a TMDb-backed movie candidate pool for the Streamlit recommender."
    )
    parser.add_argument("--output", default=TMDB_CANDIDATE_POOL_CSV)
    parser.add_argument("--max-movies-per-genre", type=int, default=50)
    parser.add_argument("--pages-per-sort", type=int, default=5)
    parser.add_argument("--min-vote-count", type=int, default=250)
    parser.add_argument("--min-vote-average", type=float, default=6.0)
    parser.add_argument("--min-release-age-days", type=int, default=30)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--tmdb-api-key", default=os.getenv("TMDB_API_KEY", ""))
    parser.add_argument(
        "--tmdb-access-token",
        default=os.getenv("TMDB_API_READ_ACCESS_TOKEN", ""),
    )
    return parser


class TmdbClient:
    def __init__(
        self,
        *,
        api_key: str,
        access_token: str,
        timeout: int,
        sleep_seconds: float,
    ) -> None:
        self.api_key = api_key.strip()
        self.access_token = access_token.strip()
        self.timeout = int(timeout)
        self.sleep_seconds = max(0.0, float(sleep_seconds))
        if not self.api_key and not self.access_token:
            raise SystemExit(
                "TMDb credentials are required. "
                "Set TMDB_API_KEY or TMDB_API_READ_ACCESS_TOKEN."
            )

    def request_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        query = dict(params)
        if self.api_key and not self.access_token:
            query["api_key"] = self.api_key

        url = f"{TMDB_API_ROOT}{path}?{urlencode(query)}"
        headers = {"accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        for attempt in range(3):
            request = Request(url, headers=headers)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = response.read().decode("utf-8")
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return json.loads(payload)
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                if exc.code == 429 and attempt < 2:
                    retry_after = exc.headers.get("Retry-After", "1")
                    try:
                        wait_seconds = max(1.0, float(retry_after))
                    except ValueError:
                        wait_seconds = 1.0
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(
                    f"TMDb request failed ({exc.code}) for {path}: {body[:300]}"
                ) from exc
            except URLError as exc:
                raise RuntimeError(f"TMDb request failed for {path}: {exc}") from exc

        raise RuntimeError(f"TMDb request failed after retries: {path}")


def fetch_tmdb_genre_index(client: TmdbClient, *, language: str) -> Dict[str, int]:
    payload = client.request_json("/genre/movie/list", {"language": language})
    genres = payload.get("genres", [])
    if not isinstance(genres, list) or not genres:
        raise RuntimeError("TMDb genre list response did not include any genres.")

    genre_index: Dict[str, int] = {}
    for item in genres:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        genre_id = item.get("id")
        if name and isinstance(genre_id, int):
            genre_index[name] = genre_id
    return genre_index


def normalize_tmdb_movie(
    movie: Dict[str, Any],
    *,
    source_tmdb_genre: str,
    sort_by: str,
) -> Dict[str, Any] | None:
    title = str(movie.get("title") or movie.get("original_title") or "").strip()
    release_date = str(movie.get("release_date") or "").strip()
    poster_path = str(movie.get("poster_path") or "").strip()
    if not title or not release_date or not poster_path:
        return None

    try:
        release_year = int(release_date[:4])
    except ValueError:
        return None

    tmdb_id = movie.get("id")
    if not isinstance(tmdb_id, int):
        return None

    return {
        "tmdb_id": tmdb_id,
        "title": title,
        "release_date": release_date,
        "release_year": release_year,
        "popularity": float(movie.get("popularity") or 0.0),
        "vote_average": float(movie.get("vote_average") or 0.0),
        "vote_count": int(movie.get("vote_count") or 0),
        "poster_url": f"{TMDB_IMAGE_BASE}{poster_path}",
        "overview": str(movie.get("overview") or "").strip(),
        "source_tmdb_genres": source_tmdb_genre,
        "source_sort_orders": sort_by,
    }


def parse_release_date(value: str) -> date | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def passes_quality_filters(
    movie: Dict[str, Any],
    *,
    min_vote_count: int,
    min_vote_average: float,
    latest_allowed_release_date: date,
) -> bool:
    release_date = parse_release_date(str(movie.get("release_date") or ""))
    if release_date is None or release_date > latest_allowed_release_date:
        return False
    if int(movie.get("vote_count") or 0) < int(min_vote_count):
        return False
    if float(movie.get("vote_average") or 0.0) < float(min_vote_average):
        return False
    return True


def fetch_ranked_tmdb_genre_pool(
    client: TmdbClient,
    *,
    tmdb_genre_name: str,
    genre_id: int,
    language: str,
    max_movies_per_genre: int,
    pages_per_sort: int,
    min_vote_count: int,
    min_vote_average: float,
    max_release_date: str,
    latest_allowed_release_date: date,
) -> List[Dict[str, Any]]:
    combined_rows: List[Dict[str, Any]] = []

    for sort_by in DISCOVER_SORT_ORDERS:
        for page in range(1, pages_per_sort + 1):
            payload = client.request_json(
                "/discover/movie",
                {
                    "include_adult": "false",
                    "include_video": "false",
                    "language": language,
                    "page": page,
                    "primary_release_date.lte": max_release_date,
                    "sort_by": sort_by,
                    "vote_average.gte": min_vote_average,
                    "vote_count.gte": min_vote_count,
                    "with_genres": genre_id,
                },
            )
            results = payload.get("results", [])
            if not isinstance(results, list) or not results:
                break

            for movie in results:
                if not isinstance(movie, dict):
                    continue
                normalized = normalize_tmdb_movie(
                    movie,
                    source_tmdb_genre=tmdb_genre_name,
                    sort_by=sort_by,
                )
                if normalized is not None and passes_quality_filters(
                    normalized,
                    min_vote_count=min_vote_count,
                    min_vote_average=min_vote_average,
                    latest_allowed_release_date=latest_allowed_release_date,
                ):
                    combined_rows.append(normalized)

    return merge_and_rank_candidates(combined_rows, limit=max_movies_per_genre)


def build_movielens_candidate_rows(
    tmdb_ranked_by_genre: Dict[str, List[Dict[str, Any]]],
    *,
    max_movies_per_genre: int,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []

    for movielens_genre, tmdb_genres in MOVIELENS_TO_TMDB_GENRES.items():
        combined_rows: List[Dict[str, Any]] = []
        for tmdb_genre in tmdb_genres:
            combined_rows.extend(tmdb_ranked_by_genre.get(tmdb_genre, []))

        ranked_rows = merge_and_rank_candidates(
            combined_rows,
            limit=max_movies_per_genre,
        )
        for row in ranked_rows:
            output_rows.append(
                {
                    "genre": movielens_genre,
                    "rank_within_genre": row["rank_within_genre"],
                    "tmdb_id": row["tmdb_id"],
                    "title": row["title"],
                    "release_year": row["release_year"],
                    "release_date": row["release_date"],
                    "popularity": row["popularity"],
                    "vote_average": row["vote_average"],
                    "vote_count": row["vote_count"],
                    "poster_url": row["poster_url"],
                    "overview": row["overview"],
                    "source_tmdb_genres": row.get("source_tmdb_genres", ""),
                    "source_sort_orders": row.get("source_sort_orders", ""),
                }
            )

    return output_rows


def save_candidate_pool(rows: Iterable[Dict[str, Any]], output_path: str) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise RuntimeError("No TMDb candidate rows were produced.")

    genre_order = {genre: index for index, genre in enumerate(MOVIELENS_TO_TMDB_GENRES.keys())}
    df["genre_order"] = df["genre"].map(genre_order).fillna(len(genre_order))
    df = df.sort_values(["genre_order", "rank_within_genre", "title"]).drop(columns=["genre_order"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    args = build_parser().parse_args()
    client = TmdbClient(
        api_key=args.tmdb_api_key,
        access_token=args.tmdb_access_token,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )

    max_release_date = date.today().isoformat()
    latest_allowed_release_date = date.today() - timedelta(days=max(0, args.min_release_age_days))
    genre_index = fetch_tmdb_genre_index(client, language=args.language)
    print(
        "[tmdb] filters: "
        f"min_vote_count={args.min_vote_count}, "
        f"min_vote_average={args.min_vote_average}, "
        f"latest_allowed_release_date={latest_allowed_release_date.isoformat()}",
        flush=True,
    )

    needed_tmdb_genres = []
    seen = set()
    for names in MOVIELENS_TO_TMDB_GENRES.values():
        for name in names:
            if name not in seen:
                seen.add(name)
                needed_tmdb_genres.append(name)

    missing_tmdb_genres = [name for name in needed_tmdb_genres if name not in genre_index]
    if missing_tmdb_genres:
        missing_text = ", ".join(missing_tmdb_genres)
        raise RuntimeError(f"TMDb genre list is missing expected genres: {missing_text}")

    tmdb_ranked_by_genre: Dict[str, List[Dict[str, Any]]] = {}
    for tmdb_genre_name in needed_tmdb_genres:
        genre_id = genre_index[tmdb_genre_name]
        print(f"[tmdb] fetching {tmdb_genre_name} ...", flush=True)
        tmdb_ranked_by_genre[tmdb_genre_name] = fetch_ranked_tmdb_genre_pool(
            client,
            tmdb_genre_name=tmdb_genre_name,
            genre_id=genre_id,
            language=args.language,
            max_movies_per_genre=args.max_movies_per_genre,
            pages_per_sort=args.pages_per_sort,
            min_vote_count=args.min_vote_count,
            min_vote_average=args.min_vote_average,
            max_release_date=max_release_date,
            latest_allowed_release_date=latest_allowed_release_date,
        )
        print(
            f"[tmdb] {tmdb_genre_name}: kept {len(tmdb_ranked_by_genre[tmdb_genre_name])}",
            flush=True,
        )

    output_rows = build_movielens_candidate_rows(
        tmdb_ranked_by_genre,
        max_movies_per_genre=args.max_movies_per_genre,
    )
    df = save_candidate_pool(output_rows, args.output)

    counts = df.groupby("genre").size().to_dict()
    print(f"[tmdb] saved {len(df)} rows to {args.output}", flush=True)
    for genre in MOVIELENS_TO_TMDB_GENRES.keys():
        print(f"  - {genre}: {counts.get(genre, 0)}", flush=True)


if __name__ == "__main__":
    main()
