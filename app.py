import streamlit as st
import numpy as np

# 直接復用你現有程式的函式
from movielens_cold_start import infer_top_genres_for_new_user, DemoEncoderSpec
from movielens_train_mf import load_preprocess_artifacts

PREPROCESS_DIR = "artifacts/preprocess"
MF_MODEL_DIR = "artifacts/mf_model"
COLD_START_DIR = "artifacts/cold_start"

st.set_page_config(page_title="Cold-start Recommender Demo", layout="wide")
st.title("Cold-start Recommender + Counterfactual Explanation (Demo)")

@st.cache_data
def load_choices():
    # 用 encoders / users_df 取出可用的 age/occupation
    _, _, _, encoders, _ = load_preprocess_artifacts(PREPROCESS_DIR)
    spec = DemoEncoderSpec.load(f"{COLD_START_DIR}/demo_encoder.json")
    ages = sorted([int(k) for k in spec.age_vocab.keys()])
    occs = sorted([int(k) for k in spec.occupation_vocab.keys()])
    genders = sorted(list(spec.gender_vocab.keys()))
    return genders, ages, occs, encoders

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

    genders, ages, occs, _ = load_choices()

    best = None  # (cost, change_desc, new_result)
    # 只改 occupation（成本 1）
    for occ2 in occs:
        if occ2 == occupation:
            continue
        r2 = run_infer(gender, age, occ2, top_m_pool, top_k_genres, top_n_movies)
        top2 = top_genres_list(r2)
        if top2 and top2[0][0] != base_top1:
            best = (1, f"occupation: {occupation} → {occ2}", r2)
            break

    # 只改 age（成本 2），只有在 occupation 找不到時才用
    if best is None:
        for age2 in ages:
            if age2 == age:
                continue
            r2 = run_infer(gender, age2, occupation, top_m_pool, top_k_genres, top_n_movies)
            top2 = top_genres_list(r2)
            if top2 and top2[0][0] != base_top1:
                best = (2, f"age: {age} → {age2}", r2)
                break

    return base, best

# Sidebar inputs
with st.sidebar:
    st.header("New user demographics")
    genders, ages, occs, encoders = load_choices()

    gender = st.selectbox("Gender", genders, index=0)
    age = st.selectbox("Age (bucket code)", ages, index=0)
    occupation = st.selectbox("Occupation (code)", occs, index=0)

    st.divider()
    st.header("Display settings")
    top_m_pool = st.slider("Top-M pooling per genre", 5, 100, 10)
    top_k_genres = st.slider("Top-K genres", 3, 20, 10)
    top_n_movies = st.slider("Top-N movies per genre", 1, 30, 5)

    run_btn = st.button("Recommend")

if run_btn:
    col1, col2 = st.columns([1, 1])

    # --- Base recommendation ---
    with col1:
        st.subheader("Recommendation")
        result = run_infer(gender, age, occupation, top_m_pool, top_k_genres, top_n_movies)
        top = top_genres_list(result)

        st.write("**Top genres**")
        st.table([{"genre": g, "score": round(s, 4)} for g, s in top])

        st.write("**Representative movies**")
        for item in result["top_genres"]:
            g = item["genre"]
            st.markdown(f"### {g}")
            rows = result["genre_top_movies"][g]
            st.table([{"movie_idx": r["movie_idx"], "score": round(r["score"], 4), "title": r["title"]} for r in rows])

    # --- Counterfactual explanation ---
    with col2:
        st.subheader("Counterfactual explanation (minimal change)")
        base, best = counterfactual_explain(gender, age, occupation, top_m_pool, top_k_genres, top_n_movies)
        base_top = top_genres_list(base)

        if best is None:
            st.info("找不到只改一個欄位就能改變 Top-1 genre 的反事實（在目前設定下）。")
        else:
            cost, change_desc, cf = best
            cf_top = top_genres_list(cf)

            st.write(f"**Minimal change:** {change_desc}  (cost={cost})")
            st.write(f"**Top-1 before:** {base_top[0][0]}  →  **Top-1 after:** {cf_top[0][0]}")

            j = jaccard_topk(base_top, cf_top, k=min(10, len(base_top), len(cf_top)))
            st.write(f"**Top-10 Jaccard similarity:** {j:.3f}  (越低代表改變越大)")

            st.write("**Before vs After (Top genres)**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Before")
                st.table([{"genre": g, "score": round(s, 4)} for g, s in base_top])
            with c2:
                st.caption("After")
                st.table([{"genre": g, "score": round(s, 4)} for g, s in cf_top])
else:
    st.write("在左側選擇 demographics，按 **Recommend** 產生推薦與反事實解釋。")