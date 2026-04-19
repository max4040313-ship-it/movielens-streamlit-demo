# movielens_cold_start.py
# ------------------------------------------------------------
# 模組三：冷啟動映射（Cold-start Mapping: Demographics → User Embedding）
#
# 目標：
# - 用訓練資料中已存在的使用者：
#     demographics (gender/age/occupation[/zip])  ->  MF 訓練得到的 user embedding U[user_idx]
#   來學一個映射： f(x_demo) -> u_hat
#
# 輸入 artifacts：
# - 模組一（preprocess）輸出：
#   users.(parquet|csv), encoders.json, meta.json
# - 模組二（mf_model）輸出：
#   U.npy, V.npy, bias_*.npy, global_mean.json, model_config.json
#
# 輸出 artifacts：
# - cold_start_model.npz：模型參數（線性回歸 / Ridge）
# - demo_encoder.json：demo encoder 的 vocab 與 feature layout
# - cold_start_config.json：訓練設定
#
# 線上推論：
# - 新用戶輸入 gender, age, occupation[, zip_prefix]
# - demo_encoder.encode(...) -> x_demo
# - cold_start_model.predict(x_demo) -> u_hat
# - 把 u_hat 丟入模組二的 GenreAggregator 做 genre 彙總
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd


# ============================================================
# 承接模組一：讀檔（parquet/csv）
# ============================================================

def _load_df(path_no_ext: str) -> pd.DataFrame:
    if os.path.exists(path_no_ext + ".parquet"):
        return pd.read_parquet(path_no_ext + ".parquet")
    if os.path.exists(path_no_ext + ".csv"):
        return pd.read_csv(path_no_ext + ".csv")
    raise FileNotFoundError(f"找不到資料檔：{path_no_ext}.parquet 或 {path_no_ext}.csv")


def load_preprocess_users(preprocess_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    讀取 users_df + encoders + meta
    users_df 預期欄位（模組一）：user_idx, gender, age, occupation, (zip, zip_prefix)
    """
    users_df = _load_df(os.path.join(preprocess_dir, "users"))
    with open(os.path.join(preprocess_dir, "encoders.json"), "r", encoding="utf-8") as f:
        encoders = json.load(f)
    with open(os.path.join(preprocess_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return users_df, encoders, meta


# ============================================================
# 承接模組二：載入 MF artifacts（僅需 U）
# ============================================================

def load_mf_user_embeddings(mf_model_dir: str) -> np.ndarray:
    """
    載入 MF 訓練完的 U.npy
    shape: [num_users, latent_dim]
    """
    path = os.path.join(mf_model_dir, "U.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 MF user embeddings：{path}")
    return np.load(path)


# ============================================================
# Demo Encoder：把 demographics 轉成固定維度 one-hot features
# ============================================================

@dataclass
class DemoEncoderSpec:
    """
    定義 demo encoder 的 layout，線上/線下必須一致。
    """
    gender_vocab: Dict[str, int]          # e.g. {"F":0,"M":1}
    age_vocab: Dict[str, int]             # JSON key 會是 str（模組一輸出 encoders.json 的特性）
    occupation_vocab: Dict[str, int]      # JSON key 會是 str
    include_zip: bool
    zip_vocab: Optional[Dict[str, int]]   # 若 include_zip=True 才有

    # feature order（固定）：[gender one-hot | age one-hot | occupation one-hot | (zip one-hot)]
    def feature_dim(self) -> int:
        d = len(self.gender_vocab) + len(self.age_vocab) + len(self.occupation_vocab)
        if self.include_zip:
            if self.zip_vocab is None:
                raise ValueError("include_zip=True 但 zip_vocab 為 None")
            d += len(self.zip_vocab)
        return d

    def save(self, out_path: str) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "DemoEncoderSpec":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return DemoEncoderSpec(**d)


class DemoEncoder:
    """
    將 (gender, age, occupation[, zip_prefix]) -> x_demo one-hot vector
    """
    def __init__(self, spec: DemoEncoderSpec):
        self.spec = spec

        # block offsets
        self.off_gender = 0
        self.off_age = self.off_gender + len(spec.gender_vocab)
        self.off_occ = self.off_age + len(spec.age_vocab)
        self.off_zip = self.off_occ + len(spec.occupation_vocab)

        self._dim = spec.feature_dim()

    @property
    def dim(self) -> int:
        return self._dim

    def encode_one(
        self,
        gender: str,
        age: int,
        occupation: int,
        zip_prefix: Optional[str] = None,
        unknown_policy: str = "zero",  # "zero" or "error"
    ) -> np.ndarray:
        """
        unknown_policy:
          - "zero": 未知類別不打任何 one-hot（向量全 0 in that block）
          - "error": 直接丟錯（線上較嚴格）
        """
        x = np.zeros(self._dim, dtype=np.float32)

        # gender
        if gender in self.spec.gender_vocab:
            x[self.off_gender + self.spec.gender_vocab[gender]] = 1.0
        elif unknown_policy == "error":
            raise ValueError(f"未知 gender={gender}")

        # age（注意：encoders.json key 是字串）
        age_key = str(int(age))
        if age_key in self.spec.age_vocab:
            x[self.off_age + self.spec.age_vocab[age_key]] = 1.0
        elif unknown_policy == "error":
            raise ValueError(f"未知 age={age}")

        # occupation
        occ_key = str(int(occupation))
        if occ_key in self.spec.occupation_vocab:
            x[self.off_occ + self.spec.occupation_vocab[occ_key]] = 1.0
        elif unknown_policy == "error":
            raise ValueError(f"未知 occupation={occupation}")

        # zip
        if self.spec.include_zip:
            if zip_prefix is None:
                if unknown_policy == "error":
                    raise ValueError("include_zip=True 但 zip_prefix=None")
            else:
                if self.spec.zip_vocab is None:
                    raise ValueError("spec.zip_vocab is None but include_zip=True")
                if zip_prefix in self.spec.zip_vocab:
                    x[self.off_zip + self.spec.zip_vocab[zip_prefix]] = 1.0
                elif unknown_policy == "error":
                    raise ValueError(f"未知 zip_prefix={zip_prefix}")

        return x

    def encode_df(
        self,
        users_df: pd.DataFrame,
        unknown_policy: str = "zero",
    ) -> np.ndarray:
        """
        將 users_df（含 gender/age/occupation[/zip_prefix]）轉成 X
        """
        need = {"gender", "age", "occupation"}
        if not need.issubset(set(users_df.columns)):
            raise ValueError(f"users_df 缺少欄位：{need - set(users_df.columns)}")

        use_zip = self.spec.include_zip
        if use_zip and ("zip_prefix" not in users_df.columns):
            # 若模組一未輸出 zip_prefix（include_zip=False），這裡會要求一致
            raise ValueError("include_zip=True 但 users_df 無 zip_prefix 欄位")

        X = np.zeros((len(users_df), self._dim), dtype=np.float32)
        for i, row in enumerate(users_df.itertuples(index=False)):
            gender = getattr(row, "gender")
            age = getattr(row, "age")
            occ = getattr(row, "occupation")
            zp = getattr(row, "zip_prefix") if use_zip else None

            X[i] = self.encode_one(
                gender=str(gender),
                age=int(age),
                occupation=int(occ),
                zip_prefix=None if (not use_zip) else str(zp),
                unknown_policy=unknown_policy
            )
        return X


# ============================================================
# Cold-start Model：多輸出回歸（Ridge / closed-form）
# ============================================================

@dataclass
class ColdStartConfig:
    """
    冷啟動映射的訓練設定。
    """
    method: str = "ridge"     # "ridge" or "ols"
    alpha: float = 10.0       # Ridge 強度（越大越保守，越不易過擬合）
    fit_intercept: bool = True
    seed: int = 42


class ColdStartModel:
    """
    線性多輸出回歸：X -> U
    - 權重 W: [d_in, d_latent]
    - 截距 b: [d_latent]（可選）
    """
    def __init__(self, W: np.ndarray, b: Optional[np.ndarray]):
        self.W = W.astype(np.float32)
        self.b = None if b is None else b.astype(np.float32)

    @property
    def latent_dim(self) -> int:
        return int(self.W.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y = X @ self.W
        if self.b is not None:
            Y = Y + self.b
        return Y

    def save_npz(self, out_path: str) -> None:
        if self.b is None:
            np.savez(out_path, W=self.W)
        else:
            np.savez(out_path, W=self.W, b=self.b)

    @staticmethod
    def load_npz(path: str) -> "ColdStartModel":
        z = np.load(path)
        W = z["W"]
        b = z["b"] if "b" in z.files else None
        return ColdStartModel(W=W, b=b)


def fit_linear_mapping_closed_form(
    X: np.ndarray,
    Y: np.ndarray,
    method: str,
    alpha: float,
    fit_intercept: bool,
) -> ColdStartModel:
    """
    不依賴 sklearn 的 closed-form 解：
      - OLS:   argmin ||XW - Y||^2
      - Ridge: argmin ||XW - Y||^2 + alpha||W||^2
    支援 fit_intercept：用增廣矩陣 [X, 1] 一次解出 W 與 b。
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    if fit_intercept:
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        Xa = np.concatenate([X, ones], axis=1)  # [n, d+1]
    else:
        Xa = X

    d = Xa.shape[1]
    I = np.eye(d, dtype=np.float64)

    if method == "ols":
        # (X^T X)^-1 X^T Y
        A = Xa.T @ Xa
        B = Xa.T @ Y
        coef = np.linalg.solve(A + 1e-8 * I, B)  # 加極小數避免奇異
    elif method == "ridge":
        # (X^T X + alpha I)^-1 X^T Y
        A = Xa.T @ Xa
        B = Xa.T @ Y
        coef = np.linalg.solve(A + alpha * I, B)
    else:
        raise ValueError(f"Unknown method: {method}")

    if fit_intercept:
        W = coef[:-1, :]
        b = coef[-1, :]
        return ColdStartModel(W=W.astype(np.float32), b=b.astype(np.float32))
    return ColdStartModel(W=coef.astype(np.float32), b=None)


def train_cold_start_mapping(
    preprocess_dir: str,
    mf_model_dir: str,
    out_dir: str,
    cfg: ColdStartConfig,
) -> None:
    """
    一站式訓練：
    - 載入 users_df/encoders/meta（模組一）
    - 載入 U（模組二）
    - 建立 demo encoder
    - 訓練 X_demo -> U
    - 存檔（encoder + model + config）
    """
    os.makedirs(out_dir, exist_ok=True)

    users_df, encoders, meta = load_preprocess_users(preprocess_dir)
    U = load_mf_user_embeddings(mf_model_dir)  # [num_users, latent_dim]

    # 基本一致性檢查
    if "user_idx" not in users_df.columns:
        raise ValueError("users_df 缺少 user_idx 欄位（模組一應已產生）")
    num_users_meta = int(meta["n_users"])
    if U.shape[0] != num_users_meta:
        raise ValueError(f"MF U.shape[0]={U.shape[0]} 與 meta[n_users]={num_users_meta} 不一致")

    # users_df 只保留可對齊 U 的 user_idx，並排序對齊
    users_df = users_df.copy()
    users_df = users_df[users_df["user_idx"].between(0, U.shape[0] - 1)]
    users_df = users_df.sort_values("user_idx").reset_index(drop=True)

    # 取出對應的 target embeddings
    Y = U[users_df["user_idx"].to_numpy(dtype=np.int64)]  # [n, latent_dim]

    # 建立 encoder spec（承接 encoders.json 的 vocab）
    spec = DemoEncoderSpec(
        gender_vocab=encoders["gender_vocab"],                # key:str
        age_vocab=encoders["age_vocab"],                      # key:str
        occupation_vocab=encoders["occupation_vocab"],        # key:str
        include_zip=bool(meta.get("include_zip", False)),
        zip_vocab=encoders.get("zip_vocab", None),
    )
    demo_encoder = DemoEncoder(spec)

    # 建 X
    X = demo_encoder.encode_df(users_df, unknown_policy="zero")  # [n, d_in]

    # 訓練（closed-form）
    model = fit_linear_mapping_closed_form(
        X=X,
        Y=Y,
        method=cfg.method,
        alpha=float(cfg.alpha),
        fit_intercept=bool(cfg.fit_intercept),
    )

    # 存檔
    spec.save(os.path.join(out_dir, "demo_encoder.json"))
    with open(os.path.join(out_dir, "cold_start_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    model.save_npz(os.path.join(out_dir, "cold_start_model.npz"))

    # 簡單訓練誤差檢視（僅作 sanity check，不是最終指標）
    Y_hat = model.predict(X).astype(np.float32)
    mse = float(np.mean((Y_hat - Y.astype(np.float32)) ** 2))
    print(f"[ColdStart] saved to: {out_dir}")
    print(f"[ColdStart] X_dim={X.shape[1]} latent_dim={Y.shape[1]}  train_mse={mse:.6f}")


# ============================================================
# 線上推論示例：demo -> u_hat -> top genres（復用模組二 GenreAggregator）
# ============================================================

def infer_top_genres_for_new_user(
    preprocess_dir: str,
    mf_model_dir: str,
    cold_start_dir: str,
    gender: str,
    age: int,
    occupation: int,
    zip_prefix: Optional[str] = None,
    top_m_pool: int = 50,
    top_k_genres: int = 5,
    top_n_movies_per_genre: int = 5,
) -> Dict[str, Any]:
    """
    回傳：
      {
        "u_hat": list[float],
        "top_genres": list[{genre, score}],
        "genre_top_movies": {genre: list[{movie_idx,title,score}]}
      }

    注意：此函式會 import 模組二中的 GenreAggregator / MFArtifacts，
    因此請確保 movielens_train_mf.py 在同一個 python path 下可被 import。
    """
    # local import to avoid hard dependency for training-only usage
    from movielens_train_mf import load_preprocess_artifacts, MFArtifacts, GenreAggregator  # noqa

    # 載入 artifacts
    _, _, movies_df, encoders, _ = load_preprocess_artifacts(preprocess_dir)
    mf = MFArtifacts.load(mf_model_dir)

    spec = DemoEncoderSpec.load(os.path.join(cold_start_dir, "demo_encoder.json"))
    demo_encoder = DemoEncoder(spec)
    cold_model = ColdStartModel.load_npz(os.path.join(cold_start_dir, "cold_start_model.npz"))

    # encode -> predict u_hat
    x = demo_encoder.encode_one(
        gender=gender,
        age=age,
        occupation=occupation,
        zip_prefix=zip_prefix,
        unknown_policy="zero"
    ).reshape(1, -1)
    u_hat = cold_model.predict(x)[0].astype(np.float32)

    # genre aggregation
    genre_vocab = encoders["genre_vocab"]
    aggregator = GenreAggregator(
        V=mf.V,
        movies_df=movies_df,
        genre_vocab=genre_vocab,
        bias_movie=mf.bias_movie,
        global_mean=mf.global_mean
    )
    genre_scores, top_genres, genre_top_movies = aggregator.aggregate(
        u=u_hat,
        top_m_pool=top_m_pool,
        top_k_genres=top_k_genres,
        top_n_movies_per_genre=top_n_movies_per_genre
    )

    return {
        "u_hat": u_hat.tolist(),
        "top_genres": [{"genre": g, "score": float(s)} for g, s in top_genres],
        "genre_top_movies": {
            g: [{"movie_idx": int(mi), "title": str(t), "score": float(sc)} for (mi, t, sc) in genre_top_movies[g]]
            for g, _ in top_genres
        }
    }


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    preprocess_dir = "artifacts/preprocess"
    mf_model_dir = "artifacts/mf_model"
    cold_start_out_dir = "artifacts/cold_start"

    cfg = ColdStartConfig(
        method="ridge",      # "ridge" or "ols"
        alpha=10.0,          # Ridge 強度：可從 1, 10, 100 試
        fit_intercept=True,
        seed=42
    )

    # 1) 訓練並存檔
    train_cold_start_mapping(
        preprocess_dir=preprocess_dir,
        mf_model_dir=mf_model_dir,
        out_dir=cold_start_out_dir,
        cfg=cfg
    )

    # 2) 線上推論示例（請自行替換 demo）
    result = infer_top_genres_for_new_user(
        preprocess_dir=preprocess_dir,
        mf_model_dir=mf_model_dir,
        cold_start_dir=cold_start_out_dir,
        gender="M",
        age=18,              # ml-1m age 是分箱碼（如 25, 35, 45...），請用資料集內的值
        occupation=3,        # ml-1m occupation 是代碼
        zip_prefix=None,
        top_m_pool=10,
        top_k_genres=10,
        top_n_movies_per_genre=30
    )

    print("\n[Cold-start] Top genres:")
    for item in result["top_genres"]:
        print(f"  - {item['genre']:15s} score={item['score']:.4f}")
