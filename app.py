import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from movielens_cold_start import DemoEncoderSpec, infer_top_genres_for_new_user
from movielens_train_mf import load_preprocess_artifacts


PREPROCESS_DIR = "artifacts/preprocess"
MF_MODEL_DIR = "artifacts/mf_model"
COLD_START_DIR = "artifacts/cold_start"

TOP_M_POOL = 50
TOP_K_GENRES = 5
TOP_N_MOVIES = 5
SCORE_LABEL = "推薦分數"

VERSION_TYPES = ["low", "medium", "high"]
VERSION_LABELS = ["A", "B", "C"]

GENDER_LABELS = {
    "F": "女",
    "M": "男",
}

OCCUPATION_LABELS = {
    0: "其他或未指定",
    1: "學術或教育工作者",
    2: "藝術家",
    3: "行政或文書工作者",
    4: "大學生或研究生",
    5: "客服人員",
    6: "醫療照護人員",
    7: "主管或經理",
    8: "農業工作者",
    9: "家務工作者",
    10: "中小學生",
    11: "法律工作者",
    12: "程式設計師",
    13: "退休人士",
    14: "業務或行銷人員",
    15: "科學家",
    16: "自由工作者",
    17: "技術人員或工程師",
    18: "技工或工匠",
    19: "待業中",
    20: "作家",
}


st.set_page_config(page_title="電影推薦研究介面", layout="wide")


@contextmanager
def section_container() -> Iterator[None]:
    """Use Streamlit's native bordered container when available."""
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
    ages = sorted([int(k) for k in spec.age_vocab.keys()])
    occupations = sorted([int(k) for k in spec.occupation_vocab.keys()])
    genders = sorted(list(spec.gender_vocab.keys()))
    return genders, ages, occupations, encoders


def init_session_state() -> None:
    defaults = {
        "page": "form",
        "user_profile": {},
        "recommendation_result": None,
        "version_order": [],
        "label_mapping": {},
        "current_version_index": 0,
        "survey_answers": {},
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
        "label_mapping",
        "current_version_index",
        "survey_answers",
        "counterfactual_result",
    ]:
        st.session_state.pop(key, None)
    st.session_state.page = "form"
    init_session_state()
    st.rerun()


def occupation_label(occupation: int) -> str:
    return OCCUPATION_LABELS.get(int(occupation), f"職業 {occupation}")


def top_genres_list(result: Dict[str, Any]) -> List[Tuple[str, float]]:
    return [(item["genre"], float(item["score"])) for item in result["top_genres"]]


def format_score(score: float) -> str:
    return f"{float(score):.1f}"


def render_score_text(score: float) -> None:
    st.write(f"{SCORE_LABEL}：**{format_score(score)}**")


@st.cache_data
def placeholder_image(key: str, height: int = 110, width: int = 180) -> np.ndarray:
    base_colors = [
        (235, 240, 243),
        (232, 238, 242),
        (237, 239, 234),
        (239, 236, 232),
        (233, 240, 238),
    ]
    accent_colors = [
        (185, 204, 211),
        (196, 207, 214),
        (190, 205, 196),
        (210, 198, 188),
        (184, 205, 202),
    ]
    color_index = sum(ord(char) for char in key) % len(base_colors)
    image = np.full((height, width, 3), base_colors[color_index], dtype=np.uint8)

    border = 4
    image[:border, :, :] = accent_colors[color_index]
    image[-border:, :, :] = accent_colors[color_index]
    image[:, :border, :] = accent_colors[color_index]
    image[:, -border:, :] = accent_colors[color_index]

    band_top = int(height * 0.64)
    image[band_top : band_top + 8, border:-border, :] = accent_colors[color_index]
    return image


def run_infer(gender: str, age: int, occupation: int) -> Dict[str, Any]:
    return infer_top_genres_for_new_user(
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


def generate_version_order() -> List[str]:
    order = VERSION_TYPES.copy()
    random.shuffle(order)
    return order


def build_label_mapping(version_order: List[str]) -> Dict[str, str]:
    return {
        label: version_type
        for label, version_type in zip(VERSION_LABELS, version_order)
    }


def render_header(interface_label: Optional[str] = None) -> None:
    if interface_label is not None:
        st.caption(f"介面 {interface_label}")
    st.title("您的電影推薦結果")
    st.caption("以下是系統根據您提供的基本資料產生的電影推薦內容。")


def render_profile_summary(profile: Dict[str, Any]) -> None:
    with section_container():
        st.subheader("基本資料")
        cols = st.columns(3)
        entries = [
            ("性別", profile["gender_label"]),
            ("年齡", str(profile["age"])),
            ("職業", profile["occupation_label"]),
        ]
        for col, (label, value) in zip(cols, entries):
            with col:
                st.caption(label)
                st.write(f"**{value}**")


def render_genre_recommendations(result: Dict[str, Any]) -> None:
    with section_container():
        st.subheader("推薦電影類型")
        genre_items = top_genres_list(result)
        cols = st.columns(len(genre_items))
        for col, (genre, score) in zip(cols, genre_items):
            with col:
                with section_container():
                    st.markdown(f"**{genre}**")
                    render_score_text(score)


def render_movie_sections(result: Dict[str, Any]) -> None:
    with section_container():
        st.subheader("推薦電影清單")
        for item in result["top_genres"]:
            genre = item["genre"]
            rows = result["genre_top_movies"].get(genre, [])

            st.markdown(f"#### {genre}")
            cols = st.columns(min(len(rows), TOP_N_MOVIES))
            for index, (col, row) in enumerate(zip(cols, rows), start=1):
                with col:
                    with section_container():
                        st.image(
                            placeholder_image(f"{genre}-{row['title']}"),
                            width="stretch",
                        )
                        st.caption(f"第 {index} 名")
                        st.markdown(f"**{row['title']}**")
                        render_score_text(row["score"])


def render_process_explanation() -> None:
    with section_container():
        st.subheader("推薦產生方式")
        st.write("系統會根據您提供的基本資料，包括性別、年齡與職業，推估您可能的電影偏好。")
        st.write(f"接著，系統會將這個偏好與電影資料進行比對，計算不同電影類型與電影項目的{SCORE_LABEL}。")
        st.write(f"{SCORE_LABEL}較高的電影類型與電影，會被排序在較前面。")


def find_counterfactual(profile: Dict[str, Any]) -> Dict[str, Any]:
    base_result = st.session_state.recommendation_result
    base_top = top_genres_list(base_result)
    base_top1 = base_top[0][0] if base_top else "無可用結果"

    _, ages, occupations, _ = load_choices()
    gender = profile["gender"]
    age = int(profile["age"])
    occupation = int(profile["occupation"])

    for next_occupation in occupations:
        next_occupation = int(next_occupation)
        if next_occupation == occupation:
            continue

        changed_result = run_infer(gender, age, next_occupation)
        changed_top = top_genres_list(changed_result)
        if changed_top and changed_top[0][0] != base_top1:
            return {
                "found": True,
                "changed_field": "職業",
                "original_profile": {
                    "gender_label": profile["gender_label"],
                    "age": age,
                    "occupation_label": profile["occupation_label"],
                },
                "changed_profile": {
                    "gender_label": profile["gender_label"],
                    "age": age,
                    "occupation_label": occupation_label(next_occupation),
                },
                "base_top1": base_top1,
                "changed_top1": changed_top[0][0],
            }

    for next_age in ages:
        next_age = int(next_age)
        if next_age == age:
            continue

        changed_result = run_infer(gender, next_age, occupation)
        changed_top = top_genres_list(changed_result)
        if changed_top and changed_top[0][0] != base_top1:
            return {
                "found": True,
                "changed_field": "年齡",
                "original_profile": {
                    "gender_label": profile["gender_label"],
                    "age": age,
                    "occupation_label": profile["occupation_label"],
                },
                "changed_profile": {
                    "gender_label": profile["gender_label"],
                    "age": next_age,
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


def render_counterfactual_explanation(profile: Dict[str, Any]) -> None:
    cf = get_counterfactual_result(profile)

    with section_container():
        st.subheader("如果條件改變，推薦會如何變化")
        st.write("系統檢查了在只改變一項基本資料的情況下，推薦結果是否會改變。")

        if not cf["found"]:
            st.write("在目前可檢查的條件中，系統沒有找到只改變一項基本資料後會改變第一名推薦類型的情況。")
            summary = pd.DataFrame(
                [
                    {"項目": "原本第一名推薦類型", "內容": cf["base_top1"]},
                    {"項目": "改變後第一名推薦類型", "內容": "未找到可改變第一名類型的單一條件"},
                ]
            )
            st.table(summary)
            return

        original = cf["original_profile"]
        changed = cf["changed_profile"]
        changed_field = cf["changed_field"]
        original_value = original["occupation_label"] if changed_field == "職業" else original["age"]
        changed_value = changed["occupation_label"] if changed_field == "職業" else changed["age"]

        st.write(
            f"在目前結果中，如果將「{changed_field}」從 {original_value} 改為 {changed_value}，"
            f"第一名推薦類型會從 {cf['base_top1']} 改為 {cf['changed_top1']}。"
        )

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**原始條件**")
            st.write(f"性別：{original['gender_label']}")
            st.write(f"年齡：{original['age']}")
            st.write(f"職業：{original['occupation_label']}")
        with cols[1]:
            st.markdown("**改變後條件**")
            st.write(f"性別：{changed['gender_label']}")
            st.write(f"年齡：{changed['age']}")
            st.write(f"職業：{changed['occupation_label']}")

        summary = pd.DataFrame(
            [
                {"項目": "原本第一名推薦類型", "內容": cf["base_top1"]},
                {"項目": "改變後第一名推薦類型", "內容": cf["changed_top1"]},
            ]
        )
        st.table(summary)


def render_explanation(version_type: str, profile: Dict[str, Any]) -> None:
    if version_type == "low":
        return
    render_process_explanation()
    if version_type == "high":
        render_counterfactual_explanation(profile)


def render_form_page() -> None:
    st.title("電影推薦研究")
    st.caption("請先填寫基本資料。完成後，您將依序觀看三個電影推薦介面。")

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
        gender_label = st.selectbox("性別", list(gender_reverse.keys()))
        age = st.selectbox("年齡", ages, index=0)
        selected_occupation_label = st.selectbox("職業", list(occupation_display.keys()), index=0)
        submitted = st.form_submit_button("開始觀看推薦介面")

    if submitted:
        gender = gender_reverse[gender_label]
        occupation = occupation_display[selected_occupation_label]
        profile = {
            "gender": gender,
            "gender_label": gender_label,
            "age": int(age),
            "occupation": int(occupation),
            "occupation_label": selected_occupation_label,
        }
        result = run_infer(
            gender=profile["gender"],
            age=profile["age"],
            occupation=profile["occupation"],
        )
        version_order = generate_version_order()

        st.session_state.user_profile = profile
        st.session_state.recommendation_result = result
        st.session_state.version_order = version_order
        st.session_state.label_mapping = build_label_mapping(version_order)
        st.session_state.current_version_index = 0
        st.session_state.survey_answers = {}
        st.session_state.counterfactual_result = None
        st.session_state.page = "recommendation"
        st.rerun()


def render_recommendation_page() -> None:
    profile = st.session_state.user_profile
    result = st.session_state.recommendation_result
    index = st.session_state.current_version_index
    version_order = st.session_state.version_order

    if not profile or result is None or not version_order:
        st.warning("尚未建立推薦資料，請先填寫基本資料。")
        if st.button("回到基本資料頁"):
            reset_experiment()
        return

    interface_label = VERSION_LABELS[index]
    version_type = version_order[index]

    render_header(interface_label)
    render_genre_recommendations(result)
    render_movie_sections(result)
    render_explanation(version_type, profile)

    st.divider()
    is_last = index >= len(version_order) - 1
    button_text = "前往問卷" if is_last else "下一個介面"
    if st.button(button_text, type="primary", key=f"recommendation_nav_{index}"):
        if is_last:
            st.session_state.page = "survey"
        else:
            st.session_state.current_version_index = index + 1
        st.rerun()


def rating_select(label: str, key: str) -> int:
    return st.slider(label, min_value=1, max_value=7, value=4, key=key)


def render_survey_page() -> None:
    st.title("問卷填答")
    st.caption("請根據剛才依序觀看的介面 A、介面 B、介面 C 進行比較與評分。")

    label_options = [f"介面 {label}" for label in VERSION_LABELS]

    with st.form("survey_form"):
        st.subheader("整體比較")
        favorite = st.radio("您最喜歡哪一個推薦介面？", label_options, horizontal=True)
        easiest = st.radio("哪一個介面最容易理解？", label_options, horizontal=True)
        trusted = st.radio("哪一個介面最值得信任？", label_options, horizontal=True)
        acceptance_pick = st.radio(
            "哪一個介面最能提升您接受推薦的意願？",
            label_options,
            horizontal=True,
        )
        future_use = st.radio(
            "若未來實際使用電影推薦系統，您最希望看到哪一個介面版本？",
            label_options,
            horizontal=True,
        )

        st.subheader("個別介面評分")
        ratings: Dict[str, Dict[str, int]] = {}
        for label in VERSION_LABELS:
            st.markdown(f"#### 介面 {label}")
            ratings[label] = {
                "understanding": rating_select(
                    "我能理解此推薦系統如何產生推薦結果。",
                    f"rating_{label}_understanding",
                ),
                "trust": rating_select(
                    "我信任此推薦系統提供的推薦結果。",
                    f"rating_{label}_trust",
                ),
                "acceptance": rating_select(
                    "我願意參考此系統的推薦結果來選擇電影。",
                    f"rating_{label}_acceptance",
                ),
            }

        comment = st.text_area("其他意見（選填）")
        submitted = st.form_submit_button("送出問卷")

    if submitted:
        st.session_state.survey_answers = {
            "favorite_interface": favorite.replace("介面 ", ""),
            "easiest_to_understand": easiest.replace("介面 ", ""),
            "most_trusted": trusted.replace("介面 ", ""),
            "most_willing_to_accept": acceptance_pick.replace("介面 ", ""),
            "preferred_for_future_use": future_use.replace("介面 ", ""),
            "ratings": ratings,
            "comment": comment,
            "version_order": st.session_state.version_order,
            "label_mapping": st.session_state.label_mapping,
        }
        st.session_state.page = "done"
        st.rerun()


def render_done_page() -> None:
    st.title("填答完成")
    st.success("感謝您的參與。您的問卷答案已暫存在本次瀏覽工作階段中。")
    st.caption("目前版本尚未寫入 CSV 或資料庫。")

    if st.button("重新開始"):
        reset_experiment()


init_session_state()

if st.session_state.page == "form":
    render_form_page()
elif st.session_state.page == "recommendation":
    render_recommendation_page()
elif st.session_state.page == "survey":
    render_survey_page()
elif st.session_state.page == "done":
    render_done_page()
else:
    reset_experiment()
