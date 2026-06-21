from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_INPUTS = [
    Path("artifacts/preprocess/movies.parquet"),
    Path("artifacts/preprocess/movies.csv"),
    Path("data/ml-1m/movies.dat"),
]
DEFAULT_OUTPUT = Path("movie_posters.csv")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
OMDB_URL = "https://www.omdbapi.com/"
IMDB_SUGGESTION_BASE = "https://v2.sg.media-imdb.com/suggestion"
TMDB_WEB_SEARCH_URL = "https://www.themoviedb.org/search/movie"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_FILE_URL = "https://en.wikipedia.org/wiki/Special:FilePath/"
USER_AGENT = "movielens-streamlit-demo/1.0 (poster lookup script)"
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
WIKIDATA_SEARCH_LIMIT = 5
WIKIPEDIA_TITLE_BATCH_SIZE = 20
WIKIPEDIA_SEARCH_LIMIT = 6
WIKIPEDIA_FALLBACK_MAX_UNRESOLVED = 3
IMDB_SUGGESTION_LIMIT = 16
IMDB_ALLOWED_QIDS = {"movie", "tvMovie", "tvSpecial", "short", "tvShort", "video"}
IMDB_ALLOWED_KINDS = {"feature", "tv movie", "tv special", "short", "video"}
MANUAL_QUERY_ALIASES = {
    "Bewegte Mann, Der": ["Maybe... Maybe Not"],
    "Bonheur, Le": ["Happiness", "Le Bonheur"],
    "Callejón de los milagros, El": ["Midaq Alley", "The Alley of Miracles"],
    "F/X 2": ["F/X2", "F/X2: The Deadly Art of Illusion"],
    "I'm the One That I Want": ["Margaret Cho: I'm the One That I Want"],
    "Land Before Time III: The Time of the Great Giving": [
        "The Land Before Time III: The Time of the Great Giving",
        "The Time of the Great Giving",
    ],
    "Light Years": ["Gandahar"],
    "Live Virgin": ["American Virgin"],
    "Marlene Dietrich: Shadow and Light": ["Marlene Dietrich: Shadows and Light"],
    "Master Ninja I": ["Master Ninja"],
    "Nosferatu a Venezia": ["Vampire in Venice", "Nosferatu in Venice"],
    "Police Story 4: Project S (Chao ji ji hua)": ["Supercop 2", "Once a Cop"],
    "Return of the Texas Chainsaw Massacre, The": [
        "Texas Chainsaw Massacre: The Next Generation",
    ],
    "Saltmen of Tibet, The": ["Die Salzmänner von Tibet"],
    "Santitos": ["Little Saints"],
    "Tashunga": ["North Star"],
}
MANUAL_POSTER_OVERRIDES = {
    "Alien Escape": {
        "poster_url": "https://media.themoviedb.org/t/p/original/f58Pev9peTv9zDzWAYY2X7GiMbI.jpg",
        "source": "manual-curated",
        "source_id": "tmdb-web:alien-escape",
        "matched_title": "Alien Escape",
    },
    "Collectionneuse, La": {
        "poster_url": "https://upload.wikimedia.org/wikipedia/en/5/58/La_Collectionneuse_%28film%29.jpg",
        "source": "manual-curated",
        "source_id": "wikipedia:La_Collectionneuse",
        "matched_title": "La Collectionneuse",
    },
    "Fantastic Night, The (La Nuit Fantastique)": {
        "poster_url": "https://upload.wikimedia.org/wikipedia/en/4/4e/Lanuitfantastique.jpg",
        "source": "manual-curated",
        "source_id": "Fantastic_Night_(1942_film)",
        "matched_title": "Fantastic Night",
    },
    "Live Virgin": {
        "poster_url": "https://upload.wikimedia.org/wikipedia/en/1/12/American_Virgin_%282000_film%29.jpg",
        "source": "manual-curated",
        "source_id": "wikipedia:American_Virgin_(1999_film)",
        "matched_title": "American Virgin",
    },
    "Loser": {
        "poster_url": "https://media.themoviedb.org/t/p/original/vKTbNXcOE7oVYJTPBFGTNVmrm04.jpg",
        "source": "manual-curated",
        "source_id": "tmdb-web:loser",
        "matched_title": "Loser",
    },
    "Marlene Dietrich: Shadow and Light": {
        "poster_url": "https://media.themoviedb.org/t/p/original/faHxYNlQ7RGDtqmaNBpFf0HKozg.jpg",
        "source": "manual-curated",
        "source_id": "tmdb-web:marlene-dietrich-shadows-and-light",
        "matched_title": "Marlene Dietrich: Shadows and Light",
    },
    "Master Ninja I": {
        "poster_url": "https://media.themoviedb.org/t/p/original/vibIJNVLni98O10P2UofYE5pyrA.jpg",
        "source": "manual-curated",
        "source_id": "tmdb-web:master-ninja",
        "matched_title": "Master Ninja",
    },
    "Open Season": {
        "poster_url": "https://media.themoviedb.org/t/p/original/kZW1DSY4w31ZNndhTQYVCVW0eL6.jpg",
        "source": "manual-curated",
        "source_id": "tmdb-web:open-season",
        "matched_title": "Open Season",
    },
    "Santitos": {
        "poster_url": "https://media.themoviedb.org/t/p/original/gPUqoCmGLwfDqiK9dVG2b8bzQMm.jpg",
        "source": "manual-curated",
        "source_id": "tmdb-web:little-saints",
        "matched_title": "Little Saints",
    },
    "Story of Xinghua, The": {
        "poster_url": "https://media.themoviedb.org/t/p/original/yKMBI1zQza96lojjlpccQNx7SBG.jpg",
        "source": "manual-curated",
        "source_id": "tmdb-web:story-of-xinghua",
        "matched_title": "The Story of Xinghua",
    },
    "Tashunga": {
        "poster_url": "https://upload.wikimedia.org/wikipedia/en/0/0f/NorthStar1996.jpg",
        "source": "manual-curated",
        "source_id": "wikipedia:North_Star_(1996_film)",
        "matched_title": "North Star",
    },
    "Ten Benny": {
        "poster_url": "https://media.themoviedb.org/t/p/original/g3XmEqaUx8otzoLEFYSayXs2BmQ.jpg",
        "source": "manual-curated",
        "source_id": "tmdb-web:ten-benny",
        "matched_title": "Ten Benny",
    },
}
TMDB_WEB_CARD_RE = re.compile(
    r'href="(?P<href>/movie/[^"]+)"'
    r'.*?<img alt="(?P<title>[^"]*)"[^>]*src="(?P<img>https://media\.themoviedb\.org/t/p/[^"]+)"'
    r'.*?<span class="release_date[^"]*">(?P<date>[^<]*)</span>',
    re.S,
)
TITLE_YEAR_RE = re.compile(r"^(?P<title>.*)\s+\((?P<year>\d{4})\)\s*$")
TRAILING_ARTICLES = (
    "The",
    "A",
    "An",
    "Le",
    "La",
    "Les",
    "L'",
    "Il",
    "Lo",
    "Gli",
    "I",
    "El",
    "Los",
    "Las",
    "Un",
    "Une",
    "Uno",
    "Una",
    "Der",
    "Die",
    "Das",
    "De",
    "Het",
)
TRAILING_ARTICLE_RE = re.compile(
    r"^(?P<title>.+),\s*(?P<article>"
    + "|".join(re.escape(article) for article in TRAILING_ARTICLES)
    + r")$",
    re.I,
)
INNER_PAREN_RE = re.compile(r"\(([^()]*)\)")
AKA_PREFIX_RE = re.compile(r"^(?:a\.?\s*k\.?\s*a\.?|aka)\s*[:.]?\s*", re.I)
INFOBOX_IMAGE_RE = re.compile(r"(?im)^\s*\|\s*(?:image|poster)\s*=\s*(.*?)\s*$")
INFOBOX_RELEASE_RE = re.compile(
    r"(?im)^\s*\|\s*(?:released|release_date)\s*=\s*(.*?)\s*$"
)
INFOBOX_TEMPLATE_YEAR_RE = re.compile(
    r"\{\{\s*(?:film date(?: and age)?|start date(?: and age)?)\s*\|\s*((?:19|20)\d{2})",
    re.I,
)
FOUR_DIGIT_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
OUTPUT_COLUMNS = [
    "movie_id_raw",
    "title",
    "movie_title",
    "year",
    "poster_url",
    "source",
    "source_id",
    "matched_title",
    "updated_at",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def string_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_int_string(value: Any) -> str:
    text = string_value(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def split_movie_title(raw_title: Any) -> tuple[str, str]:
    title = string_value(raw_title)
    match = TITLE_YEAR_RE.match(title)
    if not match:
        return title, ""
    return match.group("title").strip(), match.group("year")


def normalize_query_title(title: Any) -> str:
    text = re.sub(r"\s+", " ", string_value(title)).strip()
    match = TRAILING_ARTICLE_RE.match(text)
    if match:
        article = match.group("article").strip()
        main_title = match.group("title").strip()
        if article.endswith("'"):
            text = f"{article}{main_title}"
        else:
            text = f"{article} {main_title}"
    return text


def normalize_for_match(title: Any) -> str:
    text = normalize_query_title(title).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def move_trailing_article(text: Any) -> str:
    value = string_value(text)
    match = TRAILING_ARTICLE_RE.match(value)
    if not match:
        return value
    return f"{match.group('article')} {match.group('title')}"


def ascii_fold(text: Any) -> str:
    value = string_value(text)
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def canonical_search_title(title: Any) -> str:
    text = ascii_fold(title)
    text = re.sub(r"\s*\((?:(?:19|20)\d{2}\s+)?film\)\s*$", "", text, flags=re.I)
    text = re.sub(r"\s*\((?:19|20)\d{4}\)\s*$", "", text)
    text = move_trailing_article(text)
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_title_options(movie: Dict[str, Any]) -> List[str]:
    raw_title = string_value(movie.get("movie_title") or movie.get("title", ""))
    if not raw_title:
        return []

    options: List[str] = []
    seen = set()

    def add_option(value: Any) -> None:
        text = normalize_query_title(value)
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        options.append(text)

    base_title = re.sub(r"\([^)]*\)", "", raw_title).strip()
    add_option(base_title or raw_title)
    for piece in re.split(r"\s+/\s+|\s*;\s*", base_title):
        add_option(piece)

    for alias in INNER_PAREN_RE.findall(raw_title):
        alias = AKA_PREFIX_RE.sub("", string_value(alias)).strip(" '\"")
        if alias and not FOUR_DIGIT_YEAR_RE.fullmatch(alias):
            add_option(alias)
            for piece in re.split(r"\s+/\s+|\s*;\s*|\s*:\s*", alias):
                add_option(piece)

    for alias in MANUAL_QUERY_ALIASES.get(raw_title, []):
        add_option(alias)

    add_option(raw_title)
    return options


def find_input_path(input_path: Optional[str]) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for path in DEFAULT_INPUTS:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find a MovieLens movies file.")


def normalize_movie_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "MovieID" in df.columns:
        rename_map["MovieID"] = "movie_id_raw"
    if "Title" in df.columns:
        rename_map["Title"] = "title"
    if "movie_id" in df.columns and "movie_id_raw" not in df.columns:
        rename_map["movie_id"] = "movie_id_raw"
    df = df.rename(columns=rename_map).copy()

    if "title" not in df.columns:
        raise ValueError("Movie table must contain a title or Title column.")
    if "movie_id_raw" not in df.columns:
        df["movie_id_raw"] = ""

    movies = df[["movie_id_raw", "title"]].copy()
    movies["movie_id_raw"] = movies["movie_id_raw"].map(clean_int_string)
    movies["title"] = movies["title"].map(string_value)
    movies = movies[movies["title"] != ""]
    movies = movies.drop_duplicates(subset=["movie_id_raw", "title"])

    parsed = movies["title"].map(split_movie_title)
    movies["movie_title"] = [item[0] for item in parsed]
    movies["year"] = [item[1] for item in parsed]
    movies["query_title"] = movies["movie_title"].map(normalize_query_title)
    return movies.reset_index(drop=True)


def load_movies(input_path: Optional[str]) -> tuple[pd.DataFrame, Path]:
    path = find_input_path(input_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".dat":
        df = pd.read_csv(
            path,
            sep="::",
            names=["movie_id_raw", "title", "Genres"],
            engine="python",
            encoding="latin-1",
        )
    else:
        raise ValueError(f"Unsupported movie file type: {path.suffix}")
    return normalize_movie_columns(df), path


def movie_key(movie: Dict[str, Any]) -> str:
    movie_id = clean_int_string(movie.get("movie_id_raw", ""))
    if movie_id:
        return movie_id
    return f"title:{string_value(movie.get('title', ''))}"


def base_record(movie: Dict[str, Any]) -> Dict[str, str]:
    return {
        "movie_id_raw": clean_int_string(movie.get("movie_id_raw", "")),
        "title": string_value(movie.get("title", "")),
        "movie_title": string_value(movie.get("movie_title", "")),
        "year": clean_int_string(movie.get("year", "")),
        "poster_url": "",
        "source": "",
        "source_id": "",
        "matched_title": "",
        "updated_at": "",
    }


def load_existing_records(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}

    records: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        record = {column: string_value(row.get(column, "")) for column in OUTPUT_COLUMNS}
        key = movie_key(record)
        if key:
            records[key] = record
    return records


def save_records(
    output_path: Path,
    movies: List[Dict[str, Any]],
    records_by_key: Dict[str, Dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for movie in movies:
        key = movie_key(movie)
        record = records_by_key.get(key, base_record(movie))
        rows.append({column: record.get(column, "") for column in OUTPUT_COLUMNS})
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def request_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    retries: int = 3,
) -> Dict[str, Any]:
    if params:
        query = urlencode({key: value for key, value in params.items() if value != ""})
        url = f"{url}?{query}"

    body = None
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if data is not None:
        body = urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    for attempt in range(retries + 1):
        request = Request(url, data=body, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            should_retry = exc.code in RETRY_HTTP_CODES and attempt < retries
            if should_retry:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                if delay <= 0:
                    delay = min(60.0, 5.0 * (2**attempt))
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempt < retries:
                time.sleep(min(30.0, 5.0 * (2**attempt)))
                continue
            raise RuntimeError(f"Network error: {exc}") from exc

    raise RuntimeError("Request failed after retries.")


def request_text(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    retries: int = 3,
) -> str:
    if params:
        query = urlencode({key: value for key, value in params.items() if value != ""})
        url = f"{url}?{query}"

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "User-Agent": USER_AGENT,
    }

    for attempt in range(retries + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            should_retry = exc.code in RETRY_HTTP_CODES and attempt < retries
            if should_retry:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                if delay <= 0:
                    delay = min(60.0, 5.0 * (2**attempt))
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempt < retries:
                time.sleep(min(30.0, 5.0 * (2**attempt)))
                continue
            raise RuntimeError(f"Network error: {exc}") from exc

    raise RuntimeError("Request failed after retries.")


def parse_release_year(value: Any) -> str:
    text = string_value(value)
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return ""


def tmdb_score(result: Dict[str, Any], query_title: str, query_year: str) -> float:
    score = 0.0
    result_year = parse_release_year(result.get("release_date", ""))
    if query_year and result_year == query_year:
        score += 100
    elif query_year and result_year:
        score -= min(abs(int(query_year) - int(result_year)), 10)

    query_key = normalize_for_match(query_title)
    if normalize_for_match(result.get("title", "")) == query_key:
        score += 20
    if normalize_for_match(result.get("original_title", "")) == query_key:
        score += 10
    if result.get("poster_path"):
        score += 5
    score += float(result.get("popularity") or 0) / 100
    return score


def best_tmdb_result(
    results: List[Dict[str, Any]],
    query_title: str,
    query_year: str,
) -> Optional[Dict[str, Any]]:
    with_posters = [result for result in results if result.get("poster_path")]
    if not with_posters:
        return None
    return max(with_posters, key=lambda result: tmdb_score(result, query_title, query_year))


def fetch_tmdb(movie: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, str]:
    query_title = string_value(movie["query_title"])
    query_year = clean_int_string(movie.get("year", ""))
    params = {
        "api_key": api_key,
        "query": query_title,
        "include_adult": "false",
        "year": query_year,
    }
    data = request_json(TMDB_SEARCH_URL, params=params, timeout=timeout)
    result = best_tmdb_result(data.get("results", []), query_title, query_year)

    if result is None and query_year:
        params.pop("year", None)
        data = request_json(TMDB_SEARCH_URL, params=params, timeout=timeout)
        result = best_tmdb_result(data.get("results", []), query_title, query_year)

    if result is None:
        return {}

    return {
        "poster_url": f"{TMDB_IMAGE_BASE}{result['poster_path']}",
        "source": "tmdb",
        "source_id": string_value(result.get("id", "")),
        "matched_title": string_value(result.get("title", "")),
        "updated_at": utc_now(),
    }


def fetch_omdb(movie: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, str]:
    params = {
        "apikey": api_key,
        "t": string_value(movie["query_title"]),
        "type": "movie",
        "r": "json",
        "y": clean_int_string(movie.get("year", "")),
    }
    data = request_json(OMDB_URL, params=params, timeout=timeout)
    poster = string_value(data.get("Poster", ""))

    if (data.get("Response") != "True" or poster in {"", "N/A"}) and params.get("y"):
        params.pop("y", None)
        data = request_json(OMDB_URL, params=params, timeout=timeout)
        poster = string_value(data.get("Poster", ""))

    if data.get("Response") != "True" or poster in {"", "N/A"}:
        return {}

    return {
        "poster_url": poster,
        "source": "omdb",
        "source_id": string_value(data.get("imdbID", "")),
        "matched_title": string_value(data.get("Title", "")),
        "updated_at": utc_now(),
    }


def sparql_string(value: Any) -> str:
    text = string_value(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{text}"'


def commons_file_url(value: Any) -> str:
    text = string_value(value)
    if not text:
        return ""
    if text.startswith("http://commons.wikimedia.org/wiki/Special:FilePath/"):
        return text.replace("http://", "https://", 1)
    if text.startswith("https://commons.wikimedia.org/wiki/Special:FilePath/"):
        return text
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return "https://commons.wikimedia.org/wiki/Special:FilePath/" + quote(
        text.replace(" ", "_")
    )


def wikipedia_file_url(filename: Any) -> str:
    text = string_value(filename)
    if not text:
        return ""
    return WIKIPEDIA_FILE_URL + quote(text.replace(" ", "_"))


def article_title_from_url(article_url: Any) -> str:
    text = string_value(article_url)
    if "/wiki/" not in text:
        return ""
    return unquote(text.rsplit("/wiki/", 1)[-1]).replace("_", " ")


def page_title_key(title: Any) -> str:
    return string_value(title).replace("_", " ").casefold()


def clean_infobox_image_value(value: Any) -> str:
    text = re.sub(r"<!--.*?-->", "", string_value(value), flags=re.S).strip()
    if not text:
        return ""

    file_match = re.search(r"\[\[\s*(?:File|Image):([^|\]]+)", text, flags=re.I)
    if file_match:
        text = file_match.group(1)
    else:
        text = re.sub(r"^\s*(?:File|Image):", "", text, flags=re.I)
        text = text.split("|", 1)[0]

    text = text.strip().strip("[]")
    if not text or text.casefold().startswith(("replace", "no image")):
        return ""
    if not re.search(r"\.(jpg|jpeg|png|webp|gif)$", text, flags=re.I):
        return ""
    return text


def extract_infobox_image(wikitext: Any) -> str:
    text = string_value(wikitext)
    for match in INFOBOX_IMAGE_RE.finditer(text):
        filename = clean_infobox_image_value(match.group(1))
        if filename:
            return filename
    return ""


def extract_infobox_release_year(wikitext: Any) -> str:
    text = string_value(wikitext)
    for match in INFOBOX_RELEASE_RE.finditer(text):
        value = re.sub(r"<!--.*?-->", "", string_value(match.group(1)), flags=re.S)
        value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.I | re.S)
        value = re.sub(r"<ref[^>]*/>", "", value, flags=re.I)
        template_match = INFOBOX_TEMPLATE_YEAR_RE.search(value)
        if template_match:
            return template_match.group(1)
        year_match = FOUR_DIGIT_YEAR_RE.search(value)
        if year_match:
            return year_match.group(1)
    return ""


def revision_content(page: Dict[str, Any]) -> str:
    revisions = page.get("revisions", [])
    if not revisions:
        return ""
    revision = revisions[0]
    slots = revision.get("slots", {})
    main_slot = slots.get("main", {})
    return string_value(
        main_slot.get("*")
        or main_slot.get("content")
        or revision.get("*")
        or revision.get("content")
    )


def fetch_wikipedia_articles(
    movies: List[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    values = []
    for movie in movies:
        year = clean_int_string(movie.get("year", ""))
        year_token = year if year else "UNDEF"
        values.append(
            "("
            f"{sparql_string(movie_key(movie))} "
            f"{sparql_string(movie['query_title'])} "
            f"{year_token}"
            ")"
        )

    query = f"""
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?movieKey ?item ?itemLabel ?article ?publicationDate WHERE {{
  VALUES (?movieKey ?queryTitle ?queryYear) {{
    {' '.join(values)}
  }}
  {{
    ?item rdfs:label ?label .
    FILTER(LANG(?label) = "en")
  }}
  UNION
  {{
    ?item skos:altLabel ?label .
    FILTER(LANG(?label) = "en")
  }}
  FILTER(LCASE(STR(?label)) = LCASE(STR(?queryTitle)))
  ?item wdt:P31/wdt:P279* wd:Q11424 .
  OPTIONAL {{ ?item wdt:P577 ?publicationDate . }}
  FILTER(!BOUND(?queryYear) || !BOUND(?publicationDate) || YEAR(?publicationDate) = ?queryYear)
  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""
    data = request_json(
        WIKIDATA_SPARQL_URL,
        data={"query": query, "format": "json"},
        timeout=timeout,
    )

    articles: Dict[str, Dict[str, str]] = {}
    scores: Dict[str, int] = {}
    for binding in data.get("results", {}).get("bindings", []):
        key = string_value(binding.get("movieKey", {}).get("value", ""))
        article_url = string_value(binding.get("article", {}).get("value", ""))
        article_title = article_title_from_url(article_url)
        if not key or not article_title:
            continue

        score = 10 if binding.get("publicationDate") else 0
        if key in scores and scores[key] >= score:
            continue

        item_url = string_value(binding.get("item", {}).get("value", ""))
        articles[key] = {
            "article_title": article_title,
            "source_id": item_url.rsplit("/", 1)[-1],
            "matched_title": string_value(binding.get("itemLabel", {}).get("value", "")),
        }
        scores[key] = score

    return articles


def fetch_wikipedia_page_images(
    articles: Dict[str, Dict[str, str]],
    timeout: int,
) -> Dict[str, str]:
    title_to_keys: Dict[str, List[str]] = {}
    titles = []
    for key, article in articles.items():
        title = article["article_title"]
        titles.append(title)
        title_to_keys.setdefault(page_title_key(title), []).append(key)

    if not titles:
        return {}

    data = request_json(
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "format": "json",
            "redirects": "1",
            "prop": "revisions",
            "rvprop": "content",
            "rvsection": "0",
            "rvslots": "main",
            "titles": "|".join(titles),
        },
        timeout=timeout,
    )

    query = data.get("query", {})
    for normalized in query.get("normalized", []):
        from_key = page_title_key(normalized.get("from", ""))
        to_key = page_title_key(normalized.get("to", ""))
        if from_key in title_to_keys:
            title_to_keys.setdefault(to_key, []).extend(title_to_keys[from_key])
    for redirect in query.get("redirects", []):
        from_key = page_title_key(redirect.get("from", ""))
        to_key = page_title_key(redirect.get("to", ""))
        if from_key in title_to_keys:
            title_to_keys.setdefault(to_key, []).extend(title_to_keys[from_key])

    images: Dict[str, str] = {}
    for page in query.get("pages", {}).values():
        title = page.get("title", "")
        filename = extract_infobox_image(revision_content(page))
        if not filename:
            continue
        for key in title_to_keys.get(page_title_key(title), []):
            images[key] = filename

    return images


def extract_wikidata_claim_string(entity: Dict[str, Any], claim_id: str) -> str:
    for claim in entity.get("claims", {}).get(claim_id, []):
        datavalue = claim.get("mainsnak", {}).get("datavalue", {})
        value = datavalue.get("value")
        if isinstance(value, str):
            return value
    return ""


def extract_wikidata_year(entity: Dict[str, Any]) -> str:
    for claim in entity.get("claims", {}).get("P577", []):
        datavalue = claim.get("mainsnak", {}).get("datavalue", {})
        value = datavalue.get("value", {})
        time_value = string_value(value.get("time", ""))
        match = re.search(r"([12]\d{3})", time_value)
        if match:
            return match.group(1)
    return ""


def entity_english_names(entity: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    label = string_value(entity.get("labels", {}).get("en", {}).get("value", ""))
    if label:
        names.append(label)
    for alias in entity.get("aliases", {}).get("en", []):
        alias_value = string_value(alias.get("value", ""))
        if alias_value:
            names.append(alias_value)
    return names


def is_film_page_wikitext(wikitext: Any) -> bool:
    return "{{infobox film" in string_value(wikitext).casefold()


def wikipedia_page_match_score(
    movie: Dict[str, Any],
    page_title: str,
    release_year: str,
) -> float:
    page_norm = canonical_search_title(page_title)
    movie_year = clean_int_string(movie.get("year", ""))
    best_score = -1.0

    for option in query_title_options(movie):
        option_norm = canonical_search_title(option)
        if not option_norm or not page_norm:
            continue
        score = SequenceMatcher(None, option_norm, page_norm).ratio() * 100
        if option_norm == page_norm:
            score += 40
        elif option_norm in page_norm or page_norm in option_norm:
            score += 20
        if movie_year and release_year and movie_year == release_year:
            score += 30
        best_score = max(best_score, score)

    return best_score


def fetch_wikipedia_search_match(
    movie: Dict[str, Any],
    timeout: int,
) -> Dict[str, str]:
    candidate_titles: List[str] = []
    seen_titles = set()
    movie_year = clean_int_string(movie.get("year", ""))

    for option in query_title_options(movie):
        search_queries = [option]
        if movie_year:
            search_queries.insert(0, f'{option} "{movie_year}"')

        for search_query in search_queries:
            try:
                data = request_json(
                    WIKIPEDIA_API_URL,
                    params={
                        "action": "query",
                        "format": "json",
                        "list": "search",
                        "srsearch": search_query,
                        "srlimit": WIKIPEDIA_SEARCH_LIMIT,
                    },
                    timeout=timeout,
                )
            except Exception:
                continue

            for result in data.get("query", {}).get("search", []):
                title = string_value(result.get("title", ""))
                key = page_title_key(title)
                if title and key not in seen_titles:
                    seen_titles.add(key)
                    candidate_titles.append(title)

    if not candidate_titles:
        return {}

    best_match: Dict[str, str] = {}
    best_score = -1.0
    for title_batch in [
        candidate_titles[index : index + WIKIPEDIA_TITLE_BATCH_SIZE]
        for index in range(0, len(candidate_titles), WIKIPEDIA_TITLE_BATCH_SIZE)
    ]:
        try:
            data = request_json(
                WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "format": "json",
                    "redirects": "1",
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvsection": "0",
                    "rvslots": "main",
                    "titles": "|".join(title_batch),
                },
                timeout=timeout,
            )
        except Exception:
            continue

        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                continue
            wikitext = revision_content(page)
            if not is_film_page_wikitext(wikitext):
                continue
            filename = extract_infobox_image(wikitext)
            if not filename:
                continue
            release_year = extract_infobox_release_year(wikitext)
            if movie_year and release_year and movie_year != release_year:
                continue
            score = wikipedia_page_match_score(movie, string_value(page.get("title", "")), release_year)
            if score > best_score:
                best_score = score
                best_match = {
                    "poster_url": wikipedia_file_url(filename),
                    "source": "wikipedia-search",
                    "source_id": clean_int_string(page.get("pageid", "")),
                    "matched_title": string_value(page.get("title", "")),
                    "updated_at": utc_now(),
                }

    return best_match if best_score >= 70 else {}


def fetch_wikidata_search_match(
    movie: Dict[str, Any],
    timeout: int,
) -> Dict[str, str]:
    search_hits: Dict[str, float] = {}
    movie_year = clean_int_string(movie.get("year", ""))
    option_norms = [canonical_search_title(option) for option in query_title_options(movie)]

    for option_index, option in enumerate(query_title_options(movie)):
        try:
            data = request_json(
                WIKIDATA_API_URL,
                params={
                    "action": "wbsearchentities",
                    "format": "json",
                    "language": "en",
                    "type": "item",
                    "limit": WIKIDATA_SEARCH_LIMIT,
                    "search": option,
                },
                timeout=timeout,
            )
        except Exception:
            continue

        for rank, result in enumerate(data.get("search", []), start=1):
            qid = string_value(result.get("id", ""))
            if not qid:
                continue
            score = 100 - (rank * 5) - option_index
            search_hits[qid] = max(search_hits.get(qid, float("-inf")), score)

    if not search_hits:
        return {}

    try:
        entity_data = request_json(
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "format": "json",
                "languages": "en",
                "props": "claims|labels|aliases",
                "ids": "|".join(search_hits.keys()),
            },
            timeout=timeout,
        )
    except Exception:
        return {}

    best_match: Dict[str, str] = {}
    best_score = -1.0
    for qid, entity in entity_data.get("entities", {}).items():
        poster_filename = extract_wikidata_claim_string(entity, "P3383")
        if not poster_filename:
            continue

        release_year = extract_wikidata_year(entity)
        if movie_year and release_year and movie_year != release_year:
            continue

        entity_names = entity_english_names(entity)
        if not entity_names:
            continue

        score = search_hits.get(qid, 0.0)
        if movie_year and release_year and movie_year == release_year:
            score += 30

        best_name_score = 0.0
        for entity_name in entity_names:
            entity_norm = canonical_search_title(entity_name)
            for option_norm in option_norms:
                if not option_norm or not entity_norm:
                    continue
                name_score = SequenceMatcher(None, option_norm, entity_norm).ratio() * 100
                if option_norm == entity_norm:
                    name_score += 40
                elif option_norm in entity_norm or entity_norm in option_norm:
                    name_score += 20
                best_name_score = max(best_name_score, name_score)
        score += best_name_score

        if score > best_score:
            best_score = score
            best_match = {
                "poster_url": commons_file_url(poster_filename),
                "source": "wikidata-search",
                "source_id": qid,
                "matched_title": entity_names[0],
                "updated_at": utc_now(),
            }

    return best_match if best_score >= 90 else {}


def fetch_imdb_suggestion_match(
    movie: Dict[str, Any],
    timeout: int,
) -> Dict[str, str]:
    movie_year = clean_int_string(movie.get("year", ""))
    best_match: Dict[str, str] = {}
    best_score = -1.0

    query_variants: List[tuple[str, str, bool]] = []
    seen_queries = set()

    def add_query_variant(query_text: str, option_text: str, uses_year_hint: bool) -> None:
        cleaned = string_value(query_text).strip()
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen_queries:
            return
        seen_queries.add(key)
        query_variants.append((cleaned, option_text, uses_year_hint))

    for option in query_title_options(movie):
        title_query = option.strip()
        add_query_variant(title_query, option, False)
        ascii_title_query = ascii_fold(title_query)
        if ascii_title_query != title_query:
            add_query_variant(ascii_title_query, option, False)
        if movie_year:
            year_query = f"{option} {movie_year}".strip()
            add_query_variant(year_query, option, True)
            ascii_year_query = ascii_fold(year_query)
            if ascii_year_query != year_query:
                add_query_variant(ascii_year_query, option, True)

    for query, option, uses_year_hint in query_variants:
        first_char = re.sub(r"[^a-z0-9]", "", query.casefold())[:1] or "_"
        url = f"{IMDB_SUGGESTION_BASE}/{first_char}/{quote(query)}.json"

        try:
            data = request_json(url, timeout=timeout)
        except Exception:
            continue

        for rank, result in enumerate(data.get("d", [])[:IMDB_SUGGESTION_LIMIT], start=1):
            image_url = string_value(result.get("i", {}).get("imageUrl", ""))
            matched_title = string_value(result.get("l", ""))
            result_year = clean_int_string(result.get("y", ""))
            qid = string_value(result.get("qid", ""))
            kind = string_value(result.get("q", ""))
            imdb_id = string_value(result.get("id", ""))

            if not image_url or not matched_title or not imdb_id:
                continue
            if not imdb_id.startswith("tt"):
                continue
            if qid and qid not in IMDB_ALLOWED_QIDS and kind.casefold() not in IMDB_ALLOWED_KINDS:
                continue
            score = 100 - (rank * 5)
            option_norm = canonical_search_title(option)
            matched_norm = canonical_search_title(matched_title)
            similarity = 0.0
            exact_title = False
            title_contains = False
            if option_norm and matched_norm:
                similarity = SequenceMatcher(None, option_norm, matched_norm).ratio() * 100
                score += similarity
                if option_norm == matched_norm:
                    exact_title = True
                    score += 40
                elif option_norm in matched_norm or matched_norm in option_norm:
                    title_contains = True
                    score += 20
                elif similarity < 70:
                    continue
            if movie_year and result_year:
                year_diff = abs(int(movie_year) - int(result_year))
                if year_diff > 1 and not (exact_title or title_contains or similarity >= 92):
                    continue
                if year_diff > 5 and not (exact_title and uses_year_hint and len(option_norm) >= 12):
                    continue
                if year_diff == 0:
                    score += 30
                else:
                    score -= min(60, year_diff * 12)
            if uses_year_hint:
                score += 20

            if score > best_score:
                best_score = score
                best_match = {
                    "poster_url": image_url,
                    "source": "imdb-suggest",
                    "source_id": imdb_id,
                    "matched_title": matched_title,
                    "updated_at": utc_now(),
                }

    return best_match if best_score >= 120 else {}


def fetch_imdb_batch(
    movies: List[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    matches: Dict[str, Dict[str, str]] = {}
    for movie in movies:
        match = fetch_imdb_suggestion_match(movie, timeout)
        if match:
            matches[movie_key(movie)] = match
    return matches


def tmdb_web_poster_url(image_url: Any) -> str:
    value = string_value(image_url)
    if not value:
        return ""
    return re.sub(r"/t/p/[^/]+/", "/t/p/original/", value, count=1)


def fetch_tmdb_web_match(
    movie: Dict[str, Any],
    timeout: int,
) -> Dict[str, str]:
    movie_year = clean_int_string(movie.get("year", ""))
    best_match: Dict[str, str] = {}
    best_score = -1.0

    query_variants: List[tuple[str, str, bool]] = []
    seen_queries = set()

    def add_query_variant(query_text: str, option_text: str, uses_year_hint: bool) -> None:
        cleaned = string_value(query_text).strip()
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen_queries:
            return
        seen_queries.add(key)
        query_variants.append((cleaned, option_text, uses_year_hint))

    for option in query_title_options(movie):
        add_query_variant(option, option, False)
        ascii_option = ascii_fold(option)
        if ascii_option != option:
            add_query_variant(ascii_option, option, False)
        if movie_year:
            year_query = f"{option} {movie_year}".strip()
            add_query_variant(year_query, option, True)
            ascii_year_query = ascii_fold(year_query)
            if ascii_year_query != year_query:
                add_query_variant(ascii_year_query, option, True)

    for query, option, uses_year_hint in query_variants:
        try:
            html = request_text(TMDB_WEB_SEARCH_URL, params={"query": query}, timeout=timeout)
        except Exception:
            continue

        for match in TMDB_WEB_CARD_RE.finditer(html):
            href = string_value(match.group("href"))
            if not re.match(r"^/movie/\d+(?:-[^/]+)?$", href):
                continue
            matched_title = string_value(match.group("title"))
            poster_url = tmdb_web_poster_url(match.group("img"))
            release_text = string_value(match.group("date"))
            year_match = re.search(r"(19|20)\d{2}", release_text)
            result_year = year_match.group(0) if year_match else ""
            if not matched_title or not poster_url:
                continue

            score = 100.0
            option_norm = canonical_search_title(option)
            matched_norm = canonical_search_title(matched_title)
            similarity = 0.0
            exact_title = False
            title_contains = False
            if option_norm and matched_norm:
                similarity = SequenceMatcher(None, option_norm, matched_norm).ratio() * 100
                score += similarity
                if option_norm == matched_norm:
                    exact_title = True
                    score += 40
                elif option_norm in matched_norm or matched_norm in option_norm:
                    title_contains = True
                    score += 20
                elif similarity < 70:
                    continue

            if movie_year and result_year:
                year_diff = abs(int(movie_year) - int(result_year))
                if year_diff > 1 and not (exact_title or title_contains or similarity >= 92):
                    continue
                if year_diff > 2 and len(option_norm) < 10:
                    continue
                if year_diff > 5 and not (exact_title and uses_year_hint and len(option_norm) >= 12):
                    continue
                if year_diff == 0:
                    score += 30
                else:
                    score -= min(60, year_diff * 12)
            if uses_year_hint:
                score += 20

            if score > best_score:
                best_score = score
                best_match = {
                    "poster_url": poster_url,
                    "source": "tmdb-web",
                    "source_id": href.split("/")[-1],
                    "matched_title": matched_title,
                    "updated_at": utc_now(),
                }

    return best_match if best_score >= 110 else {}


def fetch_tmdb_web_batch(
    movies: List[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    matches: Dict[str, Dict[str, str]] = {}
    for movie in movies:
        match = fetch_tmdb_web_match(movie, timeout)
        if match:
            matches[movie_key(movie)] = match
    return matches


def wikipedia_candidate_titles(movie: Dict[str, Any]) -> List[str]:
    title = string_value(movie["query_title"])
    year = clean_int_string(movie.get("year", ""))
    candidates = [title]
    if year:
        candidates.append(f"{title} ({year} film)")
    candidates.append(f"{title} (film)")

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = page_title_key(candidate)
        if candidate and key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates


def fetch_wikipedia_direct_batch(
    movies: List[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    matches: Dict[str, Dict[str, str]] = {}
    ranks: Dict[str, int] = {}
    movie_by_key = {movie_key(movie): movie for movie in movies}

    def candidates_for_rank(title: str, year: str, rank: int) -> str:
        if rank == 0 and year:
            return f"{title} ({year} film)"
        if rank == 1:
            return title
        if rank == 2:
            return f"{title} (film)"
        return ""

    for rank in (0, 1, 2):
        remaining = [movie for movie in movies if movie_key(movie) not in matches]
        if not remaining:
            break

        title_to_candidates: Dict[str, List[tuple[str, int]]] = {}
        titles = []
        seen_titles = set()
        for movie in remaining:
            key = movie_key(movie)
            year = clean_int_string(movie.get("year", ""))
            for option_index, option_title in enumerate(query_title_options(movie)):
                title = candidates_for_rank(option_title, year, rank)
                title_key = page_title_key(title)
                if not title_key:
                    continue
                score = rank * 100 + option_index
                title_to_candidates.setdefault(title_key, []).append((key, score))
                if title_key not in seen_titles:
                    titles.append(title)
                    seen_titles.add(title_key)

        title_batches = [
            titles[index : index + WIKIPEDIA_TITLE_BATCH_SIZE]
            for index in range(0, len(titles), WIKIPEDIA_TITLE_BATCH_SIZE)
        ]
        for title_batch in title_batches:
            if not title_batch:
                continue

            try:
                data = request_json(
                    WIKIPEDIA_API_URL,
                    params={
                        "action": "query",
                        "format": "json",
                        "redirects": "1",
                        "prop": "revisions",
                        "rvprop": "content",
                        "rvsection": "0",
                        "rvslots": "main",
                        "titles": "|".join(title_batch),
                    },
                    timeout=timeout,
                )
            except Exception as exc:
                print(
                    f"wikipedia direct rank {rank} sub-batch failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            query = data.get("query", {})
            title_aliases = {key: list(value) for key, value in title_to_candidates.items()}
            for normalized in query.get("normalized", []):
                from_key = page_title_key(normalized.get("from", ""))
                to_key = page_title_key(normalized.get("to", ""))
                if from_key in title_aliases:
                    title_aliases.setdefault(to_key, []).extend(title_aliases[from_key])
            for redirect in query.get("redirects", []):
                from_key = page_title_key(redirect.get("from", ""))
                to_key = page_title_key(redirect.get("to", ""))
                if from_key in title_aliases:
                    title_aliases.setdefault(to_key, []).extend(title_aliases[from_key])

            for page in query.get("pages", {}).values():
                if "missing" in page:
                    continue
                wikitext = revision_content(page)
                filename = extract_infobox_image(wikitext)
                if not filename:
                    continue
                release_year = extract_infobox_release_year(wikitext)

                for key, candidate_rank in title_aliases.get(
                    page_title_key(page.get("title", "")),
                    [],
                ):
                    movie_year = clean_int_string(movie_by_key.get(key, {}).get("year", ""))
                    if movie_year and not release_year:
                        continue
                    if movie_year and release_year and movie_year != release_year:
                        continue
                    if key in ranks and ranks[key] <= candidate_rank:
                        continue
                    matches[key] = {
                        "poster_url": wikipedia_file_url(filename),
                        "source": "wikipedia-infobox",
                        "source_id": clean_int_string(page.get("pageid", "")),
                        "matched_title": string_value(page.get("title", "")),
                        "updated_at": utc_now(),
                    }
                    ranks[key] = candidate_rank

    return matches


def fetch_wikipedia_batch(
    movies: List[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    try:
        matches = fetch_wikipedia_direct_batch(movies, timeout)
    except Exception as exc:
        print(f"wikipedia direct batch failed: {exc}", file=sys.stderr, flush=True)
        matches = {}

    unresolved = [movie for movie in movies if movie_key(movie) not in matches]
    if not unresolved:
        return matches

    for movie in unresolved:
        try:
            match = fetch_wikipedia_search_match(movie, timeout)
        except Exception as exc:
            print(f"wikipedia search fallback failed: {exc}", file=sys.stderr, flush=True)
            match = {}
        if match:
            matches[movie_key(movie)] = match

    unresolved = [movie for movie in movies if movie_key(movie) not in matches]
    if not unresolved:
        return matches

    for movie in unresolved:
        try:
            match = fetch_imdb_suggestion_match(movie, timeout)
        except Exception as exc:
            print(f"imdb suggestion fallback failed: {exc}", file=sys.stderr, flush=True)
            match = {}
        if match:
            matches[movie_key(movie)] = match

    unresolved = [movie for movie in movies if movie_key(movie) not in matches]
    if not unresolved:
        return matches

    for movie in unresolved:
        try:
            match = fetch_tmdb_web_match(movie, timeout)
        except Exception as exc:
            print(f"tmdb web fallback failed: {exc}", file=sys.stderr, flush=True)
            match = {}
        if match:
            matches[movie_key(movie)] = match

    unresolved = [movie for movie in movies if movie_key(movie) not in matches]
    if not unresolved:
        return matches

    for movie in unresolved:
        try:
            match = fetch_wikidata_search_match(movie, timeout)
        except Exception as exc:
            print(f"wikidata search fallback failed: {exc}", file=sys.stderr, flush=True)
            match = {}
        if match:
            matches[movie_key(movie)] = match

    unresolved = [movie for movie in movies if movie_key(movie) not in matches]
    if not unresolved:
        return matches
    if len(unresolved) > WIKIPEDIA_FALLBACK_MAX_UNRESOLVED:
        return matches

    try:
        articles = fetch_wikipedia_articles(unresolved, timeout)
        images = fetch_wikipedia_page_images(articles, timeout)
    except Exception as exc:
        print(f"wikipedia fallback batch failed: {exc}", file=sys.stderr, flush=True)
        return matches

    for key, filename in images.items():
        article = articles.get(key, {})
        poster_url = wikipedia_file_url(filename)
        if not poster_url:
            continue
        matches[key] = {
            "poster_url": poster_url,
            "source": "wikipedia-infobox",
            "source_id": article.get("source_id", ""),
            "matched_title": article.get("matched_title", article.get("article_title", "")),
            "updated_at": utc_now(),
        }

    return matches


def fetch_wikidata_batch(
    movies: List[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    values = []
    for movie in movies:
        year = clean_int_string(movie.get("year", ""))
        year_token = year if year else "UNDEF"
        values.append(
            "("
            f"{sparql_string(movie_key(movie))} "
            f"{sparql_string(movie['query_title'])} "
            f"{year_token}"
            ")"
        )

    query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?movieKey ?item ?itemLabel ?publicationDate ?poster WHERE {{
  VALUES (?movieKey ?queryTitle ?queryYear) {{
    {' '.join(values)}
  }}
  {{
    ?item rdfs:label ?label .
    FILTER(LANG(?label) = "en")
  }}
  UNION
  {{
    ?item skos:altLabel ?label .
    FILTER(LANG(?label) = "en")
  }}
  FILTER(LCASE(STR(?label)) = LCASE(STR(?queryTitle)))
  ?item wdt:P31/wdt:P279* wd:Q11424 .
  OPTIONAL {{ ?item wdt:P577 ?publicationDate . }}
  FILTER(!BOUND(?queryYear) || !BOUND(?publicationDate) || YEAR(?publicationDate) = ?queryYear)
  ?item wdt:P3383 ?poster .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""
    data = request_json(
        WIKIDATA_SPARQL_URL,
        data={"query": query, "format": "json"},
        timeout=timeout,
    )

    matches: Dict[str, Dict[str, str]] = {}
    scores: Dict[str, int] = {}
    for binding in data.get("results", {}).get("bindings", []):
        key = string_value(binding.get("movieKey", {}).get("value", ""))
        poster_url = commons_file_url(binding.get("poster", {}).get("value", ""))
        if not key or not poster_url:
            continue

        score = 10 if binding.get("publicationDate") else 0
        if key in scores and scores[key] >= score:
            continue

        item_url = string_value(binding.get("item", {}).get("value", ""))
        matches[key] = {
            "poster_url": poster_url,
            "source": "wikidata:P3383",
            "source_id": item_url.rsplit("/", 1)[-1],
            "matched_title": string_value(binding.get("itemLabel", {}).get("value", "")),
            "updated_at": utc_now(),
        }
        scores[key] = score

    return matches


def chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def choose_source(args: argparse.Namespace) -> str:
    if args.source != "auto":
        return args.source
    if args.tmdb_api_key or os.getenv("TMDB_API_KEY"):
        return "tmdb"
    if args.omdb_api_key or os.getenv("OMDB_API_KEY"):
        return "omdb"
    return "wikipedia"


def apply_match(record: Dict[str, str], match: Dict[str, str]) -> None:
    for field in ("poster_url", "source", "source_id", "matched_title", "updated_at"):
        if match.get(field):
            record[field] = match[field]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build movie_posters.csv for the MovieLens Streamlit demo."
    )
    parser.add_argument("--input", help="Movie table path. Defaults to project artifacts.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument(
        "--source",
        choices=["auto", "wikipedia", "wikidata", "imdb", "tmdbweb", "tmdb", "omdb"],
        default="auto",
        help="Poster data source. auto uses TMDb/OMDb keys if present, else Wikipedia.",
    )
    parser.add_argument("--tmdb-api-key", default=os.getenv("TMDB_API_KEY", ""))
    parser.add_argument("--omdb-api-key", default=os.getenv("OMDB_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=0, help="Limit movies for testing.")
    parser.add_argument("--batch-size", type=int, default=40, help="Wikidata batch size.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between requests.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    parser.add_argument("--refresh", action="store_true", help="Refetch existing rows.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    movies_df, input_path = load_movies(args.input)
    if args.limit and args.limit > 0:
        movies_df = movies_df.head(args.limit)

    movies = movies_df.to_dict(orient="records")
    output_path = Path(args.output)
    existing_records = load_existing_records(output_path)
    records_by_key: Dict[str, Dict[str, str]] = {}
    movies_to_fetch: List[Dict[str, Any]] = []

    for movie in movies:
        key = movie_key(movie)
        record = base_record(movie)
        if key in existing_records:
            record.update(existing_records[key])
        override = MANUAL_POSTER_OVERRIDES.get(string_value(movie.get("movie_title", "")))
        if override and not record.get("poster_url"):
            record.update({**override, "updated_at": utc_now()})
        records_by_key[key] = record
        if args.refresh or not record.get("poster_url"):
            movies_to_fetch.append(movie)

    source = choose_source(args)
    if source == "tmdb" and not args.tmdb_api_key:
        raise SystemExit("TMDB API key required. Set TMDB_API_KEY or pass --tmdb-api-key.")
    if source == "omdb" and not args.omdb_api_key:
        raise SystemExit("OMDB API key required. Set OMDB_API_KEY or pass --omdb-api-key.")

    print(f"Loaded {len(movies)} movies from {input_path}", flush=True)
    print(f"Using source: {source}", flush=True)
    print(f"Need to fetch: {len(movies_to_fetch)}", flush=True)

    if source in {"wikipedia", "wikidata", "imdb", "tmdbweb"}:
        batches = chunked(movies_to_fetch, max(args.batch_size, 1))
        for batch_index, batch in enumerate(batches, start=1):
            try:
                if source == "wikipedia":
                    matches = fetch_wikipedia_batch(batch, args.timeout)
                elif source == "imdb":
                    matches = fetch_imdb_batch(batch, args.timeout)
                elif source == "tmdbweb":
                    matches = fetch_tmdb_web_batch(batch, args.timeout)
                else:
                    matches = fetch_wikidata_batch(batch, args.timeout)
            except Exception as exc:
                print(f"{source} batch {batch_index} failed: {exc}", file=sys.stderr, flush=True)
                if args.stop_on_error:
                    raise
                matches = {}

            for movie in batch:
                key = movie_key(movie)
                if key in matches:
                    apply_match(records_by_key[key], matches[key])

            save_records(output_path, movies, records_by_key)
            fetched_count = sum(1 for record in records_by_key.values() if record.get("poster_url"))
            print(
                f"Batch {batch_index}/{len(batches)} saved. Posters: {fetched_count}",
                flush=True,
            )
            time.sleep(args.sleep)
    else:
        for index, movie in enumerate(movies_to_fetch, start=1):
            key = movie_key(movie)
            try:
                if source == "tmdb":
                    match = fetch_tmdb(movie, args.tmdb_api_key, args.timeout)
                else:
                    match = fetch_omdb(movie, args.omdb_api_key, args.timeout)
            except Exception as exc:
                print(f"{movie['title']} failed: {exc}", file=sys.stderr, flush=True)
                if args.stop_on_error:
                    raise
                match = {}

            if match:
                apply_match(records_by_key[key], match)

            if index % 20 == 0 or index == len(movies_to_fetch):
                save_records(output_path, movies, records_by_key)
                fetched_count = sum(
                    1 for record in records_by_key.values() if record.get("poster_url")
                )
                print(
                    f"{index}/{len(movies_to_fetch)} saved. Posters: {fetched_count}",
                    flush=True,
                )
            time.sleep(args.sleep)

    save_records(output_path, movies, records_by_key)
    final_count = sum(1 for record in records_by_key.values() if record.get("poster_url"))
    print(f"Done. Wrote {output_path} with {final_count} poster URLs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
