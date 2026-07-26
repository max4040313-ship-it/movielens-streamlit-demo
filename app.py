from contextlib import contextmanager
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st

from movielens_cold_start import DemoEncoderSpec, infer_top_genres_for_new_user
from movielens_train_mf import load_preprocess_artifacts
from tmdb_candidate_pool import (
    TMDB_CANDIDATE_POOL_CSV,
    attach_tmdb_candidates,
    candidate_pool_file_mtime,
    load_tmdb_candidate_pool,
)


PREPROCESS_DIR = "artifacts/preprocess"
MF_MODEL_DIR = "artifacts/mf_model"
COLD_START_DIR = "artifacts/cold_start"
POSTERS_CSV = "movie_posters.csv"
POSTER_ASPECT_RATIO = "2 / 3"

TOP_M_POOL = 50
TOP_K_GENRES = 5
TOP_N_MOVIES = 5
VERSION_TYPES = ["low", "medium", "high"]
VERSION_LABELS = [
    "\u4f4e\u900f\u660e\u5ea6",
    "\u4e2d\u900f\u660e\u5ea6",
    "\u9ad8\u900f\u660e\u5ea6",
]

GENDER_LABELS = {
    "F": "\u5973\u6027",
    "M": "\u7537\u6027",
}

OCCUPATION_LABELS = {
    0: "\u5176\u4ed6\u6216\u672a\u586b\u5beb",
    1: "\u5b78\u8853\u6559\u80b2\u4eba\u54e1",
    2: "\u85dd\u8853\u5de5\u4f5c\u8005",
    3: "\u884c\u653f\u7ba1\u7406\u4eba\u54e1",
    4: "\u5927\u5b78\u751f\u6216\u7814\u7a76\u751f",
    5: "\u5ba2\u670d\u6216\u670d\u52d9\u696d",
    6: "\u91ab\u7642\u4fdd\u5065\u4eba\u54e1",
    7: "\u4e3b\u7ba1\u6216\u7d93\u7406\u4eba",
    8: "\u8fb2\u6f01\u7267\u5f9e\u696d\u4eba\u54e1",
    9: "\u5bb6\u5ead\u7167\u9867\u8005",
    10: "\u4e2d\u5c0f\u5b78\u751f",
    11: "\u5f8b\u5e2b",
    12: "\u7a0b\u5f0f\u8a2d\u8a08\u6216\u5de5\u7a0b\u6280\u8853\u4eba\u54e1",
    13: "\u9000\u4f11",
    14: "\u696d\u52d9\u6216\u884c\u92b7\u4eba\u54e1",
    15: "\u79d1\u5b78\u7814\u7a76\u4eba\u54e1",
    16: "\u81ea\u50f1\u8005\u6216\u81ea\u7531\u5de5\u4f5c\u8005",
    17: "\u6280\u5e2b\u6216\u5de5\u5320",
    18: "\u5f85\u696d\u4e2d",
    19: "\u4f5c\u5bb6",
    20: "\u5176\u4ed6\u5c08\u696d\u4eba\u54e1",
}


st.set_page_config(
    page_title="\u96fb\u5f71\u63a8\u85a6\u7814\u7a76\u4ecb\u9762",
    layout="wide",
)


@contextmanager
def section_container() -> Iterator[None]:
    try:
        container = st.container(border=True)
    except TypeError:
        container = st.container()
    with container:
        yield


@st.cache_data
def load_choices() -> Tuple[List[str], List[int], List[int], Dict[str, Any]]:
    _, _, _, encoders, _ = load_preprocess_artifacts(PREPROCESS_DIR)
    spec = DemoEncoderSpec.load(f"{COLD_START_DIR}/demo_encoder.json")
    ages = sorted(int(k) for k in spec.age_vocab.keys())
    occupations = sorted(int(k) for k in spec.occupation_vocab.keys())
    genders = sorted(list(spec.gender_vocab.keys()))
    return genders, ages, occupations, encoders


def map_age_to_model_bucket(age: int, supported_ages: List[int]) -> int:
    age_buckets = sorted(int(value) for value in supported_ages)
    bucket = age_buckets[0]
    for candidate in age_buckets:
        if age >= candidate:
            bucket = candidate
        else:
            break
    return bucket


def format_age_display(age: int, model_age: int) -> str:
    del model_age
    return str(int(age))


def init_session_state() -> None:
    defaults = {
        "page": "form",
        "user_profile": {},
        "recommendation_result": None,
        "version_order": [],
        "current_version_index": 0,
        "counterfactual_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_experiment() -> None:
    for key in [
        "user_profile",
        "recommendation_result",
        "version_order",
        "current_version_index",
        "counterfactual_result",
    ]:
        st.session_state.pop(key, None)
    st.session_state.page = "form"
    init_session_state()
    st.rerun()


def occupation_label(occupation: int) -> str:
    return OCCUPATION_LABELS.get(int(occupation), f"\u8077\u696d {occupation}")


def top_genres_list(result: Dict[str, Any]) -> List[Tuple[str, float]]:
    return [(item["genre"], float(item["score"])) for item in result["top_genres"]]


def format_score(score: float) -> str:
    return f"{float(score):.1f}"


def render_score_text(score: float) -> None:
    st.write(
        f"\u5efa\u8b70\u5206\u6578\uff1a**{format_score(score)}**"
    )


def poster_lookup_key(title: Any) -> str:
    if title is None or pd.isna(title):
        return ""
    return " ".join(str(title).strip().casefold().split())


def poster_file_mtime(path: str) -> float:
    poster_path = Path(path)
    return poster_path.stat().st_mtime if poster_path.exists() else 0.0


def render_poster_image(poster_url: str, title: str) -> None:
    safe_url = escape(poster_url, quote=True)
    safe_title = escape(title, quote=True)
    st.markdown(
        f"""
        <div style="width: 100%; aspect-ratio: {POSTER_ASPECT_RATIO}; overflow: hidden; border-radius: 0.5rem;">
            <img
                src="{safe_url}"
                alt="{safe_title}"
                style="width: 100%; height: 100%; object-fit: cover; display: block;"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_poster_slot() -> None:
    st.markdown(
        f'<div style="width: 100%; aspect-ratio: {POSTER_ASPECT_RATIO};"></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_movie_posters(path: str, modified_time: float) -> Dict[str, str]:
    del modified_time
    poster_path = Path(path)
    if not poster_path.exists():
        return {}

    try:
        posters_df = pd.read_csv(poster_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}

    if "poster_url" not in posters_df.columns:
        return {}

    posters: Dict[str, str] = {}
    for _, row in posters_df.iterrows():
        poster_url = row.get("poster_url", "")
        if pd.isna(poster_url) or not str(poster_url).strip():
            continue
        poster_url = str(poster_url).strip()

        for title_col in ("title", "movie_title"):
            if title_col not in posters_df.columns:
                continue
            key = poster_lookup_key(row.get(title_col, ""))
            if key:
                posters.setdefault(key, poster_url)

        if "movie_title" in posters_df.columns and "year" in posters_df.columns:
            movie_title = row.get("movie_title", "")
            year = row.get("year", "")
            if not pd.isna(movie_title) and not pd.isna(year):
                year_text = str(year).strip()
                if year_text.endswith(".0"):
                    year_text = year_text[:-2]
                key = poster_lookup_key(f"{movie_title} ({year_text})")
                if key:
                    posters.setdefault(key, poster_url)

    return posters


@st.cache_data
def load_tmdb_candidates(path: str, modified_time: float) -> Dict[str, List[Dict[str, Any]]]:
    return load_tmdb_candidate_pool(path=path, modified_time=modified_time)


def run_infer(gender: str, age: int, occupation: int) -> Dict[str, Any]:
    base_result = infer_top_genres_for_new_user(
        preprocess_dir=PREPROCESS_DIR,
        mf_model_dir=MF_MODEL_DIR,
        cold_start_dir=COLD_START_DIR,
        gender=gender,
        age=age,
        occupation=occupation,
        zip_prefix=None,
        top_m_pool=TOP_M_POOL,
        top_k_genres=TOP_K_GENRES,
        top_n_movies_per_genre=TOP_N_MOVIES,
    )
    candidate_pool = load_tmdb_candidates(
        TMDB_CANDIDATE_POOL_CSV,
        candidate_pool_file_mtime(TMDB_CANDIDATE_POOL_CSV),
    )
    return attach_tmdb_candidates(
        base_result,
        candidate_pool,
        top_n_movies_per_genre=TOP_N_MOVIES,
    )


def generate_version_order() -> List[str]:
    return VERSION_TYPES.copy()


def render_header(interface_label: Optional[str] = None) -> None:
    st.title("\u60a8\u7684\u96fb\u5f71\u63a8\u85a6\u7d50\u679c")
    st.caption(
        "\u4ee5\u4e0b\u662f\u7cfb\u7d71\u6839\u64da\u60a8\u63d0\u4f9b\u7684\u57fa\u672c\u8cc7\u6599\u6240\u7522\u751f\u7684\u96fb\u5f71\u63a8\u85a6\u5167\u5bb9\u3002"
    )


def render_profile_summary(profile: Dict[str, Any]) -> None:
    with section_container():
        st.subheader("\u57fa\u672c\u8cc7\u6599")
        cols = st.columns(3)
        entries = [
            ("\u6027\u5225", profile["gender_label"]),
            ("\u5e74\u9f61", profile.get("age_display", str(profile["age"]))),
            ("\u8077\u696d", profile["occupation_label"]),
        ]
        for col, (label, value) in zip(cols, entries):
            with col:
                st.caption(label)
                st.write(f"**{value}**")


def render_genre_recommendations(result: Dict[str, Any]) -> None:
    with section_container():
        st.subheader("\u96fb\u5f71\u63a8\u85a6\u985e\u578b")
        genre_items = top_genres_list(result)
        cols = st.columns(len(genre_items))
        for col, (genre, score) in zip(cols, genre_items):
            with col:
                with section_container():
                    st.markdown(f"**{genre}**")
                    render_score_text(score)


def render_movie_sections(result: Dict[str, Any]) -> None:
    with section_container():
        st.subheader("\u96fb\u5f71\u63a8\u85a6\u6e05\u55ae")
        posters = load_movie_posters(POSTERS_CSV, poster_file_mtime(POSTERS_CSV))
        for item in result["top_genres"]:
            genre = item["genre"]
            rows = result["genre_top_movies"].get(genre, [])

            st.markdown(f"#### {genre}")
            if not rows:
                st.caption("\u76ee\u524d\u6c92\u6709\u53ef\u986f\u793a\u7684\u5019\u9078\u96fb\u5f71\u3002")
                continue

            cols = st.columns(min(len(rows), TOP_N_MOVIES))
            for movie_index, (col, row) in enumerate(zip(cols, rows), start=1):
                with col:
                    with section_container():
                        title = str(row["title"])
                        poster_url = str(row.get("poster_url", "")).strip()
                        if not poster_url:
                            poster_url = posters.get(poster_lookup_key(title), "")
                        if poster_url:
                            render_poster_image(poster_url, title)
                        else:
                            render_empty_poster_slot()
                        st.caption(f"Top {movie_index}")
                        st.markdown(f"**{title}**")
                        render_score_text(row["score"])


def render_process_explanation() -> None:
    with section_container():
        st.subheader("\u63a8\u85a6\u6d41\u7a0b\u8aaa\u660e")
        st.write(
            "\u7cfb\u7d71\u6703\u5148\u6839\u64da\u60a8\u7684\u6027\u5225\u3001\u5e74\u9f61\u8207\u8077\u696d\uff0c\u63a8\u4f30\u60a8\u53ef\u80fd\u504f\u597d\u7684\u96fb\u5f71\u985e\u578b\u3002"
        )
        st.write(
            "\u63a5\u8457\uff0c\u7cfb\u7d71\u6703\u53c3\u8003\u539f\u672c\u7684\u51b7\u555f\u52d5\u6a21\u578b\u7522\u751f top genres\uff0c\u518d\u5f9e\u65b0\u7684 TMDb \u5019\u9078\u96fb\u5f71\u4e2d\u6311\u9078\u4ee3\u8868\u4f5c\u54c1\u3002"
        )
        st.write(
            "TMDb \u5019\u9078\u96fb\u5f71\u6703\u512a\u5148\u4f9d\u7167\u5e74\u4efd\u8f03\u65b0\u3001popularity \u8f03\u9ad8\u3001vote_average \u8f03\u9ad8\u7684\u689d\u4ef6\u9032\u884c\u7be9\u9078\u3002"
        )


def find_counterfactual(profile: Dict[str, Any]) -> Dict[str, Any]:
    base_result = st.session_state.recommendation_result
    base_top = top_genres_list(base_result)
    base_top1 = base_top[0][0] if base_top else "\u76ee\u524d\u7121\u63a8\u85a6\u985e\u578b"

    _, ages, occupations, _ = load_choices()
    gender = profile["gender"]
    age = int(profile["age"])
    age_model = int(profile.get("age_model", age))
    occupation = int(profile["occupation"])

    for next_occupation in occupations:
        next_occupation = int(next_occupation)
        if next_occupation == occupation:
            continue

        changed_result = run_infer(gender, age_model, next_occupation)
        changed_top = top_genres_list(changed_result)
        if changed_top and changed_top[0][0] != base_top1:
            return {
                "found": True,
                "changed_field": "occupation",
                "original_profile": {
                    "gender_label": profile["gender_label"],
                    "age": age,
                    "age_model": age_model,
                    "age_display": profile.get("age_display", format_age_display(age, age_model)),
                    "occupation_label": profile["occupation_label"],
                },
                "changed_profile": {
                    "gender_label": profile["gender_label"],
                    "age": age,
                    "age_model": age_model,
                    "age_display": profile.get("age_display", format_age_display(age, age_model)),
                    "occupation_label": occupation_label(next_occupation),
                },
                "base_top1": base_top1,
                "changed_top1": changed_top[0][0],
            }

    for next_age in ages:
        next_age = int(next_age)
        if next_age == age_model:
            continue

        changed_result = run_infer(gender, next_age, occupation)
        changed_top = top_genres_list(changed_result)
        if changed_top and changed_top[0][0] != base_top1:
            return {
                "found": True,
                "changed_field": "age",
                "original_profile": {
                    "gender_label": profile["gender_label"],
                    "age": age,
                    "age_model": age_model,
                    "age_display": profile.get("age_display", format_age_display(age, age_model)),
                    "occupation_label": profile["occupation_label"],
                },
                "changed_profile": {
                    "gender_label": profile["gender_label"],
                    "age": next_age,
                    "age_model": next_age,
                    "age_display": format_age_display(next_age, next_age),
                    "occupation_label": profile["occupation_label"],
                },
                "base_top1": base_top1,
                "changed_top1": changed_top[0][0],
            }

    return {
        "found": False,
        "base_top1": base_top1,
    }


def get_counterfactual_result(profile: Dict[str, Any]) -> Dict[str, Any]:
    if st.session_state.counterfactual_result is None:
        st.session_state.counterfactual_result = find_counterfactual(profile)
    return st.session_state.counterfactual_result


def changed_field_label(changed_field: str) -> str:
    return "\u8077\u696d" if changed_field == "occupation" else "\u5e74\u9f61"


def render_counterfactual_explanation(profile: Dict[str, Any]) -> None:
    cf = get_counterfactual_result(profile)

    with section_container():
        st.subheader("\u53cd\u4e8b\u5be6\u89e3\u91cb")
        st.write(
            "\u9019\u500b\u5340\u584a\u7528\u4f86\u8aaa\u660e\uff1a\u5982\u679c\u53ea\u6539\u8b8a\u4e00\u500b\u689d\u4ef6\uff0c\u7b2c\u4e00\u63a8\u85a6\u985e\u578b\u662f\u5426\u6703\u8ddf\u8457\u6539\u8b8a\u3002"
        )

        if not cf["found"]:
            st.write(
                "\u5728\u76ee\u524d\u53ef\u6aa2\u67e5\u7684\u55ae\u4e00\u689d\u4ef6\u8b8a\u52d5\u4e0b\uff0c\u60a8\u7684\u7b2c\u4e00\u63a8\u85a6\u985e\u578b\u7dad\u6301\u4e0d\u8b8a\uff0c\u8868\u793a\u9019\u6b21\u63a8\u85a6\u7d50\u679c\u76f8\u5c0d\u7a69\u5b9a\u3002"
            )
            summary = pd.DataFrame(
                [
                    {"\u9805\u76ee": "\u76ee\u524d\u7b2c\u4e00\u63a8\u85a6\u985e\u578b", "\u5167\u5bb9": cf["base_top1"]},
                    {
                        "\u9805\u76ee": "\u53cd\u4e8b\u5be6\u7d50\u679c",
                        "\u5167\u5bb9": "\u5728\u55ae\u4e00\u689d\u4ef6\u8b8a\u52d5\u4e0b\uff0c\u7b2c\u4e00\u63a8\u85a6\u985e\u578b\u7dad\u6301\u4e0d\u8b8a",
                    },
                ]
            )
            st.table(summary)
            return

        original = cf["original_profile"]
        changed = cf["changed_profile"]
        field_label = changed_field_label(cf["changed_field"])
        original_value = (
            original["occupation_label"]
            if cf["changed_field"] == "occupation"
            else original.get("age_display", str(original["age"]))
        )
        changed_value = (
            changed["occupation_label"]
            if cf["changed_field"] == "occupation"
            else changed.get("age_display", str(changed["age"]))
        )

        st.write(
            f"\u5982\u679c\u53ea\u6539\u8b8a\u60a8\u7684{field_label}\uff0c\u5f9e\u300c{original_value}\u300d\u8b8a\u6210\u300c{changed_value}\u300d\uff0c"
            f"\u7b2c\u4e00\u63a8\u85a6\u985e\u578b\u6703\u5f9e\u300c{cf['base_top1']}\u300d\u6539\u70ba\u300c{cf['changed_top1']}\u300d\u3002"
        )

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**\u539f\u59cb\u689d\u4ef6**")
            st.write(f"\u6027\u5225\uff1a{original['gender_label']}")
            st.write(f"\u5e74\u9f61\uff1a{original.get('age_display', original['age'])}")
            st.write(f"\u8077\u696d\uff1a{original['occupation_label']}")
        with cols[1]:
            st.markdown("**\u6539\u8b8a\u5f8c\u689d\u4ef6**")
            st.write(f"\u6027\u5225\uff1a{changed['gender_label']}")
            st.write(f"\u5e74\u9f61\uff1a{changed.get('age_display', changed['age'])}")
            st.write(f"\u8077\u696d\uff1a{changed['occupation_label']}")

        summary = pd.DataFrame(
            [
                {"\u9805\u76ee": "\u539f\u59cb\u7b2c\u4e00\u63a8\u85a6\u985e\u578b", "\u5167\u5bb9": cf["base_top1"]},
                {"\u9805\u76ee": "\u6539\u8b8a\u5f8c\u7b2c\u4e00\u63a8\u85a6\u985e\u578b", "\u5167\u5bb9": cf["changed_top1"]},
            ]
        )
        st.table(summary)


def render_high_transparency_preview_page(profile: Dict[str, Any], result: Dict[str, Any]) -> None:
    cf = get_counterfactual_result(profile)
    top_genres = top_genres_list(result)
    top_genre = top_genres[0][0] if top_genres else "\u76ee\u524d\u7121\u63a8\u85a6\u985e\u578b"

    with section_container():
        st.subheader("\u96fb\u5f71\u63a8\u85a6\u985e\u578b")
        cols = st.columns(len(top_genres))
        for col, (genre, score) in zip(cols, top_genres):
            with col:
                with section_container():
                    st.markdown(f"**{genre}**")
                    render_score_text(score)

    with section_container():
        st.subheader("\u96fb\u5f71\u63a8\u85a6\u6e05\u55ae")
        posters = load_movie_posters(POSTERS_CSV, poster_file_mtime(POSTERS_CSV))
        for item in result["top_genres"]:
            genre = item["genre"]
            rows = result["genre_top_movies"].get(genre, [])
            if genre == top_genre:
                st.markdown(
                    f"#### {genre} <span style='font-size:0.9rem;color:#6b7280;font-weight:400;'>\u76ee\u524d\u5206\u6578\u6700\u9ad8\u7684\u63a8\u85a6\u985e\u578b</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"#### {genre}")

            if not rows:
                st.caption("\u76ee\u524d\u6c92\u6709\u53ef\u986f\u793a\u7684\u5019\u9078\u96fb\u5f71\u3002")
                continue

            cols = st.columns(min(len(rows), TOP_N_MOVIES))
            for movie_index, (col, row) in enumerate(zip(cols, rows), start=1):
                with col:
                    with section_container():
                        title = str(row["title"])
                        poster_url = str(row.get("poster_url", "")).strip()
                        if not poster_url:
                            poster_url = posters.get(poster_lookup_key(title), "")
                        if poster_url:
                            render_poster_image(poster_url, title)
                        else:
                            render_empty_poster_slot()
                        st.caption(f"Top {movie_index}")
                        st.markdown(f"**{title}**")
                        render_score_text(row["score"])

    with section_container():
        st.subheader("\u63a8\u85a6\u539f\u56e0")
        st.write(
            "\u7cfb\u7d71\u6703\u5148\u6839\u64da\u60a8\u7684\u6027\u5225\u3001\u5e74\u9f61\u8207\u8077\u696d\u63a8\u4f30\u504f\u597d\uff0c\u518d\u4f9d\u64da\u5206\u6578\u627e\u51fa\u6700\u9069\u5408\u7684\u96fb\u5f71\u985e\u578b\u3002"
        )
        st.write(
            "\u63a5\u8457\u518d\u5f9e\u8f03\u65b0\u7684 TMDb \u96fb\u5f71\u5019\u9078\u6c60\u4e2d\uff0c\u6311\u51fa\u5404\u985e\u578b\u4e0b\u6700\u5177\u4ee3\u8868\u6027\u7684\u96fb\u5f71\u4f5c\u70ba\u5c55\u793a\u7d50\u679c\u3002"
        )

        with st.expander("\u67e5\u770b\u5b8c\u6574\u53cd\u4e8b\u5be6\u89e3\u91cb", expanded=False):
            if not cf["found"]:
                st.write(
                    "\u5728\u76ee\u524d\u53ef\u6aa2\u67e5\u7684\u55ae\u4e00\u689d\u4ef6\u8b8a\u52d5\u4e0b\uff0c\u60a8\u7684\u7b2c\u4e00\u63a8\u85a6\u985e\u578b\u7dad\u6301\u4e0d\u8b8a\uff0c\u8868\u793a\u9019\u6b21\u63a8\u85a6\u7d50\u679c\u76f8\u5c0d\u7a69\u5b9a\u3002"
                )
            else:
                original = cf["original_profile"]
                changed = cf["changed_profile"]
                field_label = changed_field_label(cf["changed_field"])
                original_value = (
                    original["occupation_label"]
                    if cf["changed_field"] == "occupation"
                    else original.get("age_display", str(original["age"]))
                )
                changed_value = (
                    changed["occupation_label"]
                    if cf["changed_field"] == "occupation"
                    else changed.get("age_display", str(changed["age"]))
                )
                st.write(
                    f"\u82e5\u53ea\u6539\u8b8a\u60a8\u7684{field_label}\uff0c\u5f9e\u300c{original_value}\u300d\u8b8a\u6210\u300c{changed_value}\u300d\uff0c"
                    f"\u7b2c\u4e00\u63a8\u85a6\u985e\u578b\u6703\u5f9e\u300c{cf['base_top1']}\u300d\u6539\u70ba\u300c{cf['changed_top1']}\u300d\u3002"
                )

                detail_cols = st.columns(2)
                with detail_cols[0]:
                    st.markdown("**\u539f\u59cb\u689d\u4ef6**")
                    st.write(f"\u6027\u5225\uff1a{original['gender_label']}")
                    st.write(f"\u5e74\u9f61\uff1a{original.get('age_display', original['age'])}")
                    st.write(f"\u8077\u696d\uff1a{original['occupation_label']}")
                with detail_cols[1]:
                    st.markdown("**\u6539\u8b8a\u5f8c\u689d\u4ef6**")
                    st.write(f"\u6027\u5225\uff1a{changed['gender_label']}")
                    st.write(f"\u5e74\u9f61\uff1a{changed.get('age_display', changed['age'])}")
                    st.write(f"\u8077\u696d\uff1a{changed['occupation_label']}")


def render_explanation(version_type: str, profile: Dict[str, Any]) -> None:
    if version_type == "low":
        return
    render_process_explanation()
    if version_type == "high":
        render_counterfactual_explanation(profile)


def render_form_page() -> None:
    st.title("\u96fb\u5f71\u63a8\u85a6\u7814\u7a76")
    st.caption(
        "\u8acb\u5148\u586b\u5beb\u57fa\u672c\u8cc7\u6599\u3002\u5b8c\u6210\u5f8c\uff0c\u60a8\u5c07\u4f9d\u5e8f\u89c0\u770b\u4f4e\u900f\u660e\u5ea6\u3001\u4e2d\u900f\u660e\u5ea6\u8207\u9ad8\u900f\u660e\u5ea6\u4ecb\u9762\u3002"
    )

    genders, ages, occupations, _ = load_choices()
    gender_options = [code for code in ["F", "M"] if code in genders] or genders
    gender_reverse = {GENDER_LABELS.get(code, code): code for code in gender_options}

    occupation_options = [
        occ for occ in occupations if int(occ) in OCCUPATION_LABELS
    ] or occupations
    occupation_display = {
        occupation_label(int(occ)): int(occ)
        for occ in occupation_options
    }

    with st.form("profile_form"):
        gender_label = st.selectbox("\u6027\u5225", list(gender_reverse.keys()))
        age_text = st.text_input(
            "\u5e74\u9f61",
            value="",
            placeholder="\u8acb\u8f38\u5165\u5e74\u9f61",
        )
        selected_occupation_label = st.selectbox(
            "\u8077\u696d",
            list(occupation_display.keys()),
            index=0,
        )
        submitted = st.form_submit_button("\u7522\u751f\u63a8\u85a6\u7d50\u679c")

    if submitted:
        age_text = age_text.strip()
        if not age_text:
            st.warning("\u8acb\u8f38\u5165\u5e74\u9f61\u3002")
            return
        if not age_text.isdigit():
            st.warning("\u5e74\u9f61\u8acb\u8f38\u5165\u6578\u5b57\u3002")
            return

        age = int(age_text)
        if age < 1 or age > 120:
            st.warning("\u5e74\u9f61\u8acb\u8f38\u5165 1 \u5230 120 \u4e4b\u9593\u7684\u6578\u5b57\u3002")
            return

        age_model = map_age_to_model_bucket(age, ages)
        gender = gender_reverse[gender_label]
        occupation = occupation_display[selected_occupation_label]
        profile = {
            "gender": gender,
            "gender_label": gender_label,
            "age": age,
            "age_model": age_model,
            "age_display": format_age_display(age, age_model),
            "occupation": int(occupation),
            "occupation_label": selected_occupation_label,
        }

        try:
            result = run_infer(
                gender=profile["gender"],
                age=profile["age_model"],
                occupation=profile["occupation"],
            )
        except (FileNotFoundError, ValueError) as exc:
            st.error(
                "\u627e\u4e0d\u5230 TMDb \u5019\u9078\u96fb\u5f71\u8cc7\u6599\uff0c\u8acb\u5148\u57f7\u884c `python build_tmdb_candidate_pool.py`\uff0c"
                "\u4e26\u78ba\u8a8d\u5df2\u8a2d\u5b9a `TMDB_API_KEY` \u6216 `TMDB_API_READ_ACCESS_TOKEN`\u3002"
            )
            st.caption(str(exc))
            return

        st.session_state.user_profile = profile
        st.session_state.recommendation_result = result
        st.session_state.version_order = generate_version_order()
        st.session_state.current_version_index = 0
        st.session_state.counterfactual_result = None
        st.session_state.page = "recommendation"
        st.rerun()


def render_recommendation_page() -> None:
    profile = st.session_state.user_profile
    result = st.session_state.recommendation_result
    index = st.session_state.current_version_index
    version_order = st.session_state.version_order
    high_preview_index = len(version_order)

    if not profile or result is None or not version_order:
        st.warning(
            "\u63a8\u85a6\u8cc7\u6599\u4e0d\u5b58\u5728\uff0c\u8acb\u91cd\u65b0\u56de\u5230\u524d\u4e00\u9801\u586b\u5beb\u57fa\u672c\u8cc7\u6599\u3002"
        )
        if st.button("\u8fd4\u56de\u57fa\u672c\u8cc7\u6599\u9801"):
            reset_experiment()
        return

    if index < len(version_order):
        interface_label = VERSION_LABELS[index]
        version_type = version_order[index]
        render_header(interface_label)
        render_profile_summary(profile)
        render_genre_recommendations(result)
        render_movie_sections(result)
        render_explanation(version_type, profile)
    else:
        render_header("\u9ad8\u900f\u660e\u5ea6\u8ffd\u52a0\u9801")
        render_profile_summary(profile)
        render_high_transparency_preview_page(profile, result)

    st.divider()
    is_last = index >= high_preview_index
    button_text = "\u5b8c\u6210" if is_last else "\u4e0b\u4e00\u9801"
    if st.button(button_text, type="primary", key=f"recommendation_nav_{index}"):
        if is_last:
            st.session_state.page = "done"
        else:
            st.session_state.current_version_index = index + 1
        st.rerun()


def render_done_page() -> None:
    st.title("\u63a8\u85a6\u700f\u89bd\u5b8c\u6210")
    st.success("\u60a8\u5df2\u5b8c\u6210\u6240\u6709\u7248\u672c\u7684\u63a8\u85a6\u9801\u9762\u700f\u89bd\u3002")
    st.caption(
        "\u5982\u679c\u60a8\u8981\u7e7c\u7e8c\u6bd4\u8f03\u4e0d\u540c\u900f\u660e\u5ea6\u7248\u672c\uff0c\u6216\u91cd\u65b0\u7522\u751f\u65b0\u7684\u63a8\u85a6\u7d50\u679c\uff0c\u53ef\u4ee5\u91cd\u65b0\u958b\u59cb\u3002"
    )
    if st.button("\u91cd\u65b0\u958b\u59cb"):
        reset_experiment()


init_session_state()

if st.session_state.page == "form":
    render_form_page()
elif st.session_state.page == "recommendation":
    render_recommendation_page()
elif st.session_state.page == "done":
    render_done_page()
else:
    reset_experiment()
