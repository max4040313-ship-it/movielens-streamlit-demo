import streamlit as st
import numpy as np

# 直接復用你現有程式的函式
from movielens_cold_start import infer_top_genres_for_new_user, DemoEncoderSpec
from movielens_train_mf import load_preprocess_artifacts

PREPROCESS_DIR = "artifacts/preprocess"
MF_MODEL_DIR = "artifacts/mf_model"
COLD_START_DIR = "artifacts/cold_start"

st.set_page_config(page_title="冷啟動推薦系統示範", layout="wide")

# ---------------------------
# Session state 初始化
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "form"

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# ---------------------------
# 載入可選項
# ---------------------------
@st.cache_data
def load_choices():
    _, _, _, encoders, _ = load_preprocess_artifacts(PREPROCESS_DIR)
    spec = DemoEncoderSpec.load(f"{COLD_START_DIR}/demo_encoder.json")
    ages = sorted([int(k) for k in spec.age_vocab.keys()])
    occs = sorted([int(k) for k in spec.occupation_vocab.keys()])
    genders = sorted(list(spec.gender_vocab.keys()))
    return genders, ages, occs, encoders

# ---------------------------
# 工具函式
# ---------------------------
def top_genres_list(result):
    return [(x["genre"], float(x["score"])) for x in result["top_genres"]]

def jaccard_topk(a, b, k=10):
    sa = set([g for g, _ in a[:k]])
    sb = set([g for g, _ in b[:k]])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def run_infer(gender, age, occupation, top_m_pool, top_k_genres, top_n_movies):
    return infer_top_genres_for_new_user(
        preprocess_dir=PREPROCESS_DIR,
        mf_model_dir=MF_MODEL_DIR,
        cold_start_dir=COLD_START_DIR,
        gender=gender,
        age=age,
        occupation=occupation,
        zip_prefix=None,
        top_m_pool=top_m_pool,
        top_k_genres=top_k_genres,
        top_n_movies_per_genre=top_n_movies,
    )

def counterfactual_explain(gender, age, occupation, top_m_pool, top_k_genres, top_n_movies):
    """
    找「最小反事實」：不改 gender，只允許改 occupation 或 age（改一個欄位）。
    目標：讓 Top-1 genre 改變。
    """
    base = run_infer(gender, age, occupation, top_m_pool, top_k_genres, top_n_movies)
    base_top = top_genres_list(base)
    base_top1 = base_top[0][0] if base_top else None

    _, ages, occs, _ = load_choices()

    best = None  # (cost, change_desc, new_result)

    # 只改 occupation（成本 1）
    for occ2 in occs:
        if occ2 == occupation:
            continue
        r2 = run_infer(gender, age, occ2, top_m_pool, top_k_genres, top_n_movies)
        top2 = top_genres_list(r2)
        if top2 and top2[0][0] != base_top1:
            best = (1, f"職業：{occupation} → {occ2}", r2)
            break

    # 只改 age（成本 2），只有在 occupation 找不到時才用
    if best is None:
        for age2 in ages:
            if age2 == age:
                continue
            r2 = run_infer(gender, age2, occupation, top_m_pool, top_k_genres, top_n_movies)
            top2 = top_genres_list(r2)
            if top2 and top2[0][0] != base_top1:
                best = (2, f"年齡：{age} → {age2}", r2)
                break

    return base, best

# ---------------------------
# 第一頁：填寫基本資料
# ---------------------------
if st.session_state.page == "form":
    st.title("新用戶您好")
    st.subheader("請先填寫基本資料")

    genders, ages, occs, encoders = load_choices()

    # 顯示中文，實際傳給模型仍然是 F / M
    gender_map = {
        "女": "F",
        "男": "M"
    }

    gender_label = st.selectbox("性別", list(gender_map.keys()), index=0)
    gender = gender_map[gender_label]

    age = st.selectbox("年齡", ages, index=0)
    occupation = st.selectbox("職業）", occs, index=0)

    st.subheader("透明度選擇")
    transparency_level = st.selectbox(
        "透明度條件",
        ["低透明度", "中透明度", "高透明度"],
        index=0
    )

    st.subheader("顯示設定")
    top_m_pool = st.slider("每個類型取前 M 部電影", 5, 100, 10)
    top_k_genres = st.slider("前 K 個推薦類型", 3, 20, 10)
    top_n_movies = st.slider("每個類型顯示前 N 部電影", 1, 30, 5)

    if st.button("登入"):
        st.session_state.user_profile = {
            "gender": gender,
            "age": age,
            "occupation": occupation,
            "transparency_level": transparency_level,
            "top_m_pool": top_m_pool,
            "top_k_genres": top_k_genres,
            "top_n_movies": top_n_movies,
        }
        st.session_state.page = "result"
        st.rerun()

# ---------------------------
# 第二頁：推薦結果頁
# ---------------------------
elif st.session_state.page == "result":
    profile = st.session_state.user_profile

    gender = profile["gender"]
    age = profile["age"]
    occupation = profile["occupation"]
    transparency_level = profile["transparency_level"]
    top_m_pool = profile["top_m_pool"]
    top_k_genres = profile["top_k_genres"]
    top_n_movies = profile["top_n_movies"]

    gender_text = "女" if gender == "F" else "男"

    st.title("推薦結果")

    if st.button("返回上一頁"):
        st.session_state.page = "form"
        st.rerun()

    col1, col2 = st.columns([1, 1])

    # ---------------------------
    # 左邊：推薦結果
    # ---------------------------
    with col1:
        st.subheader("推薦內容")

        result = run_infer(
            gender, age, occupation,
            top_m_pool, top_k_genres, top_n_movies
        )
        top = top_genres_list(result)

        st.write("**推薦類型**")
        st.table([{"類型": g, "分數": round(s, 4)} for g, s in top])

        st.write("**代表電影**")
        for item in result["top_genres"]:
            g = item["genre"]
            st.markdown(f"### {g}")
            rows = result["genre_top_movies"][g]
            st.table([
                {
                    "電影編號": r["movie_idx"],
                    "分數": round(r["score"], 4),
                    "片名": r["title"]
                }
                for r in rows
            ])

    # ---------------------------
    # 右邊：解釋區塊
    # ---------------------------
    with col2:
        st.subheader("解釋說明")

        if transparency_level == "低透明度":
            st.info("此條件僅顯示推薦結果，不提供額外解釋。")

        elif transparency_level == "中透明度":
            st.write("**程序解釋**")
            st.write(
                f"系統根據使用者的人口統計資料（性別={gender_text}、年齡={age}、職業={occupation}）"
                "預測其偏好向量，並依此產生推薦類型與代表電影。"
            )

        elif transparency_level == "高透明度":
            st.write("**程序解釋**")
            st.write(
                f"系統根據使用者的人口統計資料（性別={gender_text}、年齡={age}、職業={occupation}）"
                "預測其偏好向量，並依此產生推薦類型與代表電影。"
            )

            st.write("**反事實解釋**")
            base, best = counterfactual_explain(
                gender, age, occupation,
                top_m_pool, top_k_genres, top_n_movies
            )

            if best is None:
                st.warning("找不到可改變第一推薦類型的單一人口特徵反事實。")
            else:
                cost, change_desc, new_result = best

                base_top = top_genres_list(base)
                new_top = top_genres_list(new_result)

                base_top1 = base_top[0][0] if base_top else "N/A"
                new_top1 = new_top[0][0] if new_top else "N/A"
                jacc = jaccard_topk(base_top, new_top, k=top_k_genres)

                st.write(f"最小改動：{change_desc}")
                st.write(f"改動成本：{cost}")
                st.write(f"原始第一推薦類型：{base_top1}")
                st.write(f"反事實第一推薦類型：{new_top1}")
                st.write(f"賈卡德相似係數：{jacc:.4f}")

                st.write("**反事實後推薦類型**")
                st.table([{"類型": g, "分數": round(s, 4)} for g, s in new_top])