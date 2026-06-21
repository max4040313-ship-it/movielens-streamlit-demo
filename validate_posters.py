from __future__ import annotations

import argparse
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

from fetch_posters import (
    FOUR_DIGIT_YEAR_RE,
    TRAILING_ARTICLE_RE,
    WIKIPEDIA_API_URL,
    clean_int_string,
    extract_infobox_image,
    extract_infobox_release_year,
    page_title_key,
    request_json,
    revision_content,
    split_movie_title,
    string_value,
    wikipedia_file_url,
)


FILM_SUFFIX_RE = re.compile(r"\s*\((?:(?:19|20)\d{2}\s+)?film\)\s*$", re.I)
NUMBER_TOKEN_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
ROMAN_TOKEN_MAP = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Second-pass validation for movie_posters.csv."
    )
    parser.add_argument("--input", default="movie_posters.csv")
    parser.add_argument("--report", default="movie_posters_validation.csv")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--clear-invalid", action="store_true")
    parser.add_argument("--wikipedia-only", action="store_true")
    parser.add_argument("--suspicious-only", action="store_true")
    return parser.parse_args()


def ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def strip_parenthetical_content(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def move_trailing_article(text: str) -> str:
    match = TRAILING_ARTICLE_RE.match(text.strip())
    if not match:
        return text
    return f"{match.group('article')} {match.group('title')}"


def canonical_title(text: str) -> str:
    text = string_value(text)
    text = FILM_SUFFIX_RE.sub("", text)
    text = re.sub(r"\s*\((?:19|20)\d{4}\)\s*$", "", text)
    text = move_trailing_article(text)
    text = ascii_fold(text).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9()]+", " ", text)
    tokens = []
    for token in text.split():
        token = NUMBER_TOKEN_MAP.get(token, token)
        token = ROMAN_TOKEN_MAP.get(token, token)
        tokens.append(token)
    return " ".join(tokens).strip()


def title_variants(title: str) -> Set[str]:
    raw = string_value(title)
    variants = {canonical_title(raw)}

    no_paren = canonical_title(strip_parenthetical_content(raw))
    if no_paren:
        variants.add(no_paren)

    main_title, _ = split_movie_title(raw)
    main_title = string_value(main_title)
    if main_title:
        variants.add(canonical_title(main_title))
        variants.add(canonical_title(strip_parenthetical_content(main_title)))

    split_candidates = re.split(r"\s*:\s*|\s+-\s+", raw)
    for candidate in split_candidates:
        candidate = string_value(candidate)
        if candidate:
            variants.add(canonical_title(candidate))
            variants.add(canonical_title(strip_parenthetical_content(candidate)))

    return {variant for variant in variants if variant}


def titles_match(expected_title: str, matched_title: str) -> bool:
    expected_variants = title_variants(expected_title)
    matched_variants = title_variants(matched_title)
    if not expected_variants or not matched_variants:
        return False

    if expected_variants & matched_variants:
        return True

    best_ratio = 0.0
    for expected in expected_variants:
        for matched in matched_variants:
            ratio = SequenceMatcher(None, expected, matched).ratio()
            best_ratio = max(best_ratio, ratio)
            if ratio >= 0.9:
                return True
    return best_ratio >= 0.82


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def fetch_page_metadata(
    matched_titles: List[str],
    batch_size: int,
    timeout: int,
) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    title_lookup = {page_title_key(title): title for title in matched_titles}

    for batch in chunked(matched_titles, batch_size):
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
                "titles": "|".join(batch),
            },
            timeout=timeout,
        )

        query = data.get("query", {})
        alias_map: Dict[str, Set[str]] = {}
        for title in batch:
            alias_map.setdefault(page_title_key(title), set()).add(title)

        for normalized in query.get("normalized", []):
            from_key = page_title_key(normalized.get("from", ""))
            to_key = page_title_key(normalized.get("to", ""))
            alias_map.setdefault(to_key, set()).update(alias_map.get(from_key, set()))
        for redirect in query.get("redirects", []):
            from_key = page_title_key(redirect.get("from", ""))
            to_key = page_title_key(redirect.get("to", ""))
            alias_map.setdefault(to_key, set()).update(alias_map.get(from_key, set()))

        for page in query.get("pages", {}).values():
            if "missing" in page:
                continue
            page_title = string_value(page.get("title", ""))
            page_key = page_title_key(page_title)
            wikitext = revision_content(page)
            filename = extract_infobox_image(wikitext)
            release_year = extract_infobox_release_year(wikitext)
            record = {
                "page_title": page_title,
                "release_year": release_year,
                "poster_url": wikipedia_file_url(filename) if filename else "",
                "is_film_page": "{{infobox film" in wikitext.casefold(),
            }
            for alias in alias_map.get(page_key, set()):
                metadata[alias] = record

    return metadata


def validation_reason(row: pd.Series, metadata: Dict[str, str]) -> str:
    matched_title = string_value(row.get("matched_title", ""))
    if not matched_title:
        return "missing_matched_title"

    page_title = metadata.get("page_title", "")
    if not page_title:
        return "page_not_found"

    release_year = clean_int_string(metadata.get("release_year", ""))
    movie_year = clean_int_string(row.get("year", ""))
    if movie_year and release_year and movie_year != release_year:
        return "release_year_mismatch"

    candidate_titles = [string_value(row.get("title", "")), string_value(row.get("movie_title", ""))]
    if not any(titles_match(candidate, page_title) for candidate in candidate_titles if candidate):
        return "title_mismatch"

    page_poster = string_value(metadata.get("poster_url", ""))
    stored_poster = string_value(row.get("poster_url", ""))
    if page_poster and stored_poster and page_poster != stored_poster:
        return "poster_url_changed"

    if not page_poster:
        return "missing_page_image"

    return ""


def looks_suspicious(row: pd.Series) -> bool:
    poster_url = string_value(row.get("poster_url", ""))
    matched_title = string_value(row.get("matched_title", ""))
    if not poster_url:
        return False
    if not matched_title:
        return True

    explicit_year_match = FOUR_DIGIT_YEAR_RE.search(matched_title)
    movie_year = clean_int_string(row.get("year", ""))
    if explicit_year_match and movie_year and explicit_year_match.group(1) != movie_year:
        return True

    candidate_titles = [string_value(row.get("title", "")), string_value(row.get("movie_title", ""))]
    return not any(titles_match(candidate, matched_title) for candidate in candidate_titles if candidate)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    report_path = Path(args.report)

    df = pd.read_csv(input_path).fillna("")
    if args.wikipedia_only:
        mask = df["source"].astype(str).str.startswith("wikipedia")
    else:
        mask = df["poster_url"].astype(str).str.strip().ne("")
    candidates = df.loc[mask].copy()
    if args.suspicious_only:
        candidates = candidates[candidates.apply(looks_suspicious, axis=1)].copy()

    matched_titles = sorted(
        {string_value(title) for title in candidates["matched_title"] if string_value(title)}
    )
    metadata = fetch_page_metadata(matched_titles, args.batch_size, args.timeout)

    report_rows = []
    invalid_indices = []
    for index, row in candidates.iterrows():
        matched_title = string_value(row.get("matched_title", ""))
        page_metadata = metadata.get(matched_title, {})
        reason = validation_reason(row, page_metadata)
        is_valid = reason == ""
        report_rows.append(
            {
                "movie_id_raw": row.get("movie_id_raw", ""),
                "title": row.get("title", ""),
                "year": row.get("year", ""),
                "matched_title": matched_title,
                "page_title": page_metadata.get("page_title", ""),
                "page_release_year": page_metadata.get("release_year", ""),
                "stored_poster_url": row.get("poster_url", ""),
                "page_poster_url": page_metadata.get("poster_url", ""),
                "is_film_page": bool(page_metadata.get("is_film_page", False)),
                "source": row.get("source", ""),
                "is_valid": is_valid,
                "reason": reason,
            }
        )
        if not is_valid:
            invalid_indices.append(index)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False, encoding="utf-8")

    if args.clear_invalid and invalid_indices:
        clear_columns = ["poster_url", "source", "source_id", "matched_title", "updated_at"]
        for column in clear_columns:
            df.loc[invalid_indices, column] = ""
        df.to_csv(input_path, index=False, encoding="utf-8")

    print(f"validated={len(report_rows)} invalid={len(invalid_indices)} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
