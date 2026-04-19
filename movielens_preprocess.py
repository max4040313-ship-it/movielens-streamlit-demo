# movielens_preprocess.py
# ------------------------------------------------------------
# 模組一：MovieLens ml-1m 資料讀取與前處理
#
# 目標：
# 1. 將原始 :: 分隔的 .dat 檔轉成乾淨 DataFrame
# 2. 建立 user/movie 的連續整數 index（供 embedding lookup）
# 3. 處理人口統計學特徵詞彙表（gender / age / occupation）
# 4. 將電影類型 (movie genres) 轉成 multi-hot 向量
# 5. 輸出「可重現」的 artifacts，供訓練與線上推論共用
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# Configuration 與 Artifact 結構定義
# ============================================================

@dataclass
class PreprocessConfig:
    """
    控制整個前處理流程的設定檔物件。
    儲存這些設定可確保實驗的可重現性 (Reproducibility)。
    """
    data_dir: str          # 原始 MovieLens ml-1m 資料夾路徑
    output_dir: str        # 處理後結果 (artifacts) 的輸出路徑
    seed: int = 42         # 隨機種子，確保資料切分結果一致
    test_ratio: float = 0.2 # 測試集比例

    # 過濾參數：可用於移除評分過少的極端使用者或電影，維持資料品質
    min_ratings_per_user: int = 1
    min_ratings_per_movie: int = 1

    # 郵遞區號處理：由於 ml-1m 的 Zip code 非常稀疏且格式不一，預設關閉
    include_zip: bool = False
    zip_prefix_len: int = 3 # 若啟用，僅取前三碼（區域代表性較強）

    # 輸出格式設定
    output_format: str = "parquet"   # "parquet" or "csv"
    parquet_engine: str = "pyarrow"  # 或 "fastparquet"，依據環境來選擇


@dataclass
class Encoders:
    """
    存放所有從「原始值」轉換到「整數索引」的對照表 (Vocabularies)。
    這是線上推論 (Inference) 與離線訓練 (Training) 之間最重要的對接契約，
    確保兩者使用的編碼邏輯完全一致。
    """
    user_id_map: Dict[int, int]         # 原始 UserID -> 0...N-1 索引
    movie_id_map: Dict[int, int]        # 原始 MovieID -> 0...M-1 索引

    gender_vocab: Dict[str, int]        # 性別 -> 索引
    age_vocab: Dict[int, int]           # 年齡組別 -> 索引 (ml-1m 的年齡已是區段編碼)
    occupation_vocab: Dict[int, int]    # 職業代碼 -> 索引
    zip_vocab: Optional[Dict[str, int]] # 郵遞區號 -> 索引

    genre_vocab: Dict[str, int]         # 電影類型 -> 向量欄位索引

    def to_jsonable(self) -> dict:
        """
        將 Encoder 物件轉成 JSON 可儲存的格式。
        注意：JSON 的 Key 必須是字串，因此需要將 int key 轉為 str。
        """
        d = asdict(self)
        d["user_id_map"] = {str(k): v for k, v in self.user_id_map.items()}
        d["movie_id_map"] = {str(k): v for k, v in self.movie_id_map.items()}
        d["age_vocab"] = {str(k): v for k, v in self.age_vocab.items()}
        d["occupation_vocab"] = {str(k): v for k, v in self.occupation_vocab.items()}
        if d["zip_vocab"] is not None:
            d["zip_vocab"] = {str(k): v for k, v in d["zip_vocab"].items()}
        return d


# ============================================================
# 讀取 MovieLens 原始資料
# ============================================================

def _read_dat(path: str, names: List[str]) -> pd.DataFrame:
    """
    底層輔助函式：讀取 .dat 檔案。
    MovieLens ml-1m 使用 '::' 作為分隔符，需指定 python 引擎解析。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案: {path}")

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=names,
        encoding="latin-1" # ml-1m 包含特殊字元，需使用 latin-1
    )


def load_ml_1m(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    讀取 ratings, users, movies 三個原始檔案，
    並進行基本的型別轉換 (Normalization)，避免後續合併 (Join) 時發生錯誤。
    """
    ratings = _read_dat(
        os.path.join(data_dir, "ratings.dat"),
        ["UserID", "MovieID", "Rating", "Timestamp"]
    )
    users = _read_dat(
        os.path.join(data_dir, "users.dat"),
        ["UserID", "Gender", "Age", "Occupation", "Zip"]
    )
    movies = _read_dat(
        os.path.join(data_dir, "movies.dat"),
        ["MovieID", "Title", "Genres"]
    )

    # 型別正規化：確保 ID 為整數，評分為浮點數
    ratings["UserID"] = ratings["UserID"].astype(int)
    ratings["MovieID"] = ratings["MovieID"].astype(int)
    ratings["Rating"] = ratings["Rating"].astype(float)
    ratings["Timestamp"] = ratings["Timestamp"].astype(int)

    users["UserID"] = users["UserID"].astype(int)
    users["Gender"] = users["Gender"].astype(str)
    users["Age"] = users["Age"].astype(int)
    users["Occupation"] = users["Occupation"].astype(int)
    users["Zip"] = users["Zip"].astype(str)

    movies["MovieID"] = movies["MovieID"].astype(int)
    movies["Title"] = movies["Title"].astype(str)
    movies["Genres"] = movies["Genres"].astype(str)

    return ratings, users, movies


# ============================================================
# ID / Vocabulary 建立 (核心處理邏輯)
# ============================================================

def build_id_maps(ratings: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    將原始 ID 對照到從 0 開始的連續整數。
    為什麼從 ratings 建立？
    - 確保 Embedding 層只為「真正有互動過」的用戶與電影建立，節省記憶體。
    """
    user_ids = np.sort(ratings["UserID"].unique())
    movie_ids = np.sort(ratings["MovieID"].unique())

    # 建立映射表 e.g., {原始ID: 連續索引}
    user_id_map = {int(uid): i for i, uid in enumerate(user_ids)}
    movie_id_map = {int(mid): i for i, mid in enumerate(movie_ids)}

    return user_id_map, movie_id_map


def build_demo_vocabs(
    users: pd.DataFrame,
    include_zip: bool,
    zip_prefix_len: int
):
    """
    建立人口統計學特徵 (Demographics) 的詞彙表。
    將類別型特徵（如性別 M/F）轉為整數索引。
    """
    gender_vocab = {g: i for i, g in enumerate(sorted(users["gender"].unique()))}
    age_vocab = {int(a): i for i, a in enumerate(sorted(users["age"].unique()))}
    occupation_vocab = {int(o): i for i, o in enumerate(sorted(users["occupation"].unique()))}

    zip_vocab = None
    zip_feat = pd.Series([None] * len(users), index=users.index)

    # 選擇性處理郵遞區號
    if include_zip:
        # zip_feat = users["Zip"].str.slice(0, zip_prefix_len) # 只取前幾碼
        zip_feat = users["zip"].str.slice(0, zip_prefix_len) # 只取前幾碼

        zip_vocab = {z: i for i, z in enumerate(sorted(zip_feat.unique()))}

    return gender_vocab, age_vocab, occupation_vocab, zip_vocab, zip_feat


def build_genre_vocab_and_multihot(movies: pd.DataFrame):
    """
    處理電影類型。電影通常擁有多個類型（如 "動作|喜劇"）。
    1. 建立 Genre 詞彙表。
    2. 將 Genres 欄位轉為 Multi-hot 向量 (每部電影一個 0/1 陣列)。
    """
    # 將 "Action|Comedy" 轉為 ["Action", "Comedy"]
    genres_list = movies["Genres"].apply(lambda s: s.split("|") if s else [])
    # 取得所有不重複的類型並排序
    all_genres = sorted({g for gs in genres_list for g in gs})

    genre_vocab = {g: i for i, g in enumerate(all_genres)}

    # 初始化 Multi-hot 矩陣 (電影數 x 類型總數)
    multihot = np.zeros((len(movies), len(all_genres)), dtype=np.int8)
    for i, gs in enumerate(genres_list):
        for g in gs:
            multihot[i, genre_vocab[g]] = 1

    # 將 Multi-hot 矩陣轉為 DataFrame 方便後續合併
    genre_cols = [f"genre_{g}" for g in all_genres]
    multihot_df = pd.DataFrame(multihot, columns=genre_cols, index=movies.index)
    
    movies = movies.copy()
    movies["genres_list"] = genres_list

    return genre_vocab, pd.concat([movies, multihot_df], axis=1)


# ============================================================
# 資料切分
# ============================================================

def split_train_test_ratings(
    ratings_df: pd.DataFrame,
    test_ratio: float,
    seed: int
):
    """
    隨機切分訓練集與測試集。
    
    備註：這是在 Rating 層級做隨機切分。
    更進階的作法包含：
    - Leave-last-out: 每個用戶最後一筆當測試。
    - Time-based: 用時間點切分，模擬真實推論情境。
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ratings_df))
    rng.shuffle(idx)

    cut = int(len(idx) * (1 - test_ratio))
    train_idx, test_idx = idx[:cut], idx[cut:]

    return (
        ratings_df.iloc[train_idx].reset_index(drop=True),
        ratings_df.iloc[test_idx].reset_index(drop=True),
    )


# ============================================================
# 主前處理 Pipeline
# ============================================================

def preprocess_ml_1m(cfg: PreprocessConfig):
    """
    核心 Entry Point：整合所有步驟並產出最終的資料集與編碼器。
    """
    # 1. 讀取原始資料
    ratings, users, movies = load_ml_1m(cfg.data_dir)

    # 2. 建立 ID 映射 (Mapping)
    user_id_map, movie_id_map = build_id_maps(ratings)

    # 3. 處理評分表 (加入連續索引)
    ratings_df = ratings.rename(columns={
        "UserID": "user_id_raw",
        "MovieID": "movie_id_raw",
        "Rating": "rating",
        "Timestamp": "timestamp"
    })
    ratings_df["user_idx"] = ratings_df["user_id_raw"].map(user_id_map)
    ratings_df["movie_idx"] = ratings_df["movie_id_raw"].map(movie_id_map)

    # 4. 處理用戶資料表
    users_df = users.rename(columns={
        "UserID": "user_id_raw",
        "Gender": "gender",
        "Age": "age",
        "Occupation": "occupation",
        "Zip": "zip"
    })
    # 只保留出現在評分表中的用戶
    users_df = users_df[users_df["user_id_raw"].isin(user_id_map)]
    users_df["user_idx"] = users_df["user_id_raw"].map(user_id_map)

    # 建立人口統計特徵詞彙表
    gender_vocab, age_vocab, occupation_vocab, zip_vocab, zip_prefix = \
        build_demo_vocabs(users_df, cfg.include_zip, cfg.zip_prefix_len)
    
    if cfg.include_zip:
        users_df["zip_prefix"] = zip_prefix

    # 5. 處理電影資料表 (加入 Multi-hot 類型)
    movies_df = movies.rename(columns={
        "MovieID": "movie_id_raw",
        "Title": "title",
        "Genres": "genres_raw"
    })
    movies_df = movies_df[movies_df["movie_id_raw"].isin(movie_id_map)]
    movies_df["movie_idx"] = movies_df["movie_id_raw"].map(movie_id_map)

    genre_vocab, movies_df = build_genre_vocab_and_multihot(
        movies_df.rename(columns={"genres_raw": "Genres"})
    )

    # 6. 切分資料
    train_df, test_df = split_train_test_ratings(
        ratings_df, cfg.test_ratio, cfg.seed
    )

    # 7. 封裝所有編碼器
    encoders = Encoders(
        user_id_map=user_id_map,
        movie_id_map=movie_id_map,
        gender_vocab=gender_vocab,
        age_vocab=age_vocab,
        occupation_vocab=occupation_vocab,
        zip_vocab=zip_vocab,
        genre_vocab=genre_vocab
    )

    return ratings_df, users_df, movies_df, train_df, test_df, encoders

def save_artifacts(cfg: PreprocessConfig,
                   ratings_df: pd.DataFrame,
                   users_df: pd.DataFrame,
                   movies_df: pd.DataFrame,
                   train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   encoders: Encoders) -> None:
    """
    將處理完後的結果存入硬碟。
    包含 CSV 資料檔、JSON 編碼檔以及 Metadata（描述資料集規模的後設資料）。
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    fmt = cfg.output_format.lower()

    # 儲存資料
    _save_df(ratings_df, os.path.join(cfg.output_dir, "ratings"), fmt, cfg.parquet_engine)
    _save_df(users_df,   os.path.join(cfg.output_dir, "users"),   fmt, cfg.parquet_engine)
    _save_df(movies_df,  os.path.join(cfg.output_dir, "movies"),  fmt, cfg.parquet_engine)
    _save_df(train_df,   os.path.join(cfg.output_dir, "train"),   fmt, cfg.parquet_engine)
    _save_df(test_df,    os.path.join(cfg.output_dir, "test"),    fmt, cfg.parquet_engine)

    # 儲存編碼器供線上推論使用
    with open(os.path.join(cfg.output_dir, "encoders.json"), "w", encoding="utf-8") as f:
        json.dump(encoders.to_jsonable(), f, ensure_ascii=False, indent=2)

    # 儲存 Metadata，這對模型定義 (Model Definition) 時設定 Embedding 維度至關重要
    meta = {
        "seed": cfg.seed,
        "test_ratio": cfg.test_ratio,
        "n_users": int(len(encoders.user_id_map)),
        "n_movies": int(len(encoders.movie_id_map)),
        "n_genres": int(len(encoders.genre_vocab)),
        "include_zip": cfg.include_zip,
        "zip_prefix_len": cfg.zip_prefix_len if cfg.include_zip else None,
        "output_format": fmt,
    }
    with open(os.path.join(cfg.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# 儲存資料
def _save_df(df: pd.DataFrame, path_no_ext: str, fmt: str, engine: str = "pyarrow") -> None:
    if fmt == "csv":
        df.to_csv(path_no_ext + ".csv", index=False)
    elif fmt == "parquet":
        df.to_parquet(path_no_ext + ".parquet", index=False, engine=engine)
    else:
        raise ValueError(f"Unsupported output_format: {fmt}")


# ============================================================
# 主程式執行入口
# ============================================================
if __name__ == "__main__":
    # 建立設定參數
    cfg = PreprocessConfig(
        data_dir="data/ml-1m",
        output_dir="artifacts/preprocess",
        seed=42,
        test_ratio=0.2,
        include_zip=False,
        output_format="parquet",      # "parquet" or "csv"
        parquet_engine="pyarrow"
    )

    # 執行前處理
    ratings_df, users_df, movies_df, train_df, test_df, encoders = preprocess_ml_1m(cfg)
    
    # 儲存結果
    save_artifacts(cfg, ratings_df, users_df, movies_df, train_df, test_df, encoders)

    print("前處理完成！")
    print(f"評分筆數: {ratings_df.shape}")
    print(f"使用者數: {users_df.shape[0]}, 電影數: {movies_df.shape[0]}")
    print(f"訓練集/測試集大小: {train_df.shape[0]} / {test_df.shape[0]}")


# 這段程式碼的三個重點：
# 連續索引 (Indexing)：深度學習推薦系統模型（如 Matrix Factorization 或 DeepFM）通常使用 Embedding 層，這要求 UserID 與 MovieID 必須是從 $0$ 到 $N-1$ 的連續整數。
# 多類型處理 (Multi-hot)：一部電影可能同時是「喜劇」和「動作」，程式碼透過 build_genre_vocab_and_multihot 將其轉換成電腦能理解的 0/1 向量。
# 線上/線下一致性：透過 Encoders 類別將對照表存為 encoders.json，這保證了當新的請求進來時，系統會用與訓練時相同的編碼方式來處理資料。