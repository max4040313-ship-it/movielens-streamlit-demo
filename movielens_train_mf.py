# movielens_train_mf.py
# ------------------------------------------------------------
# 模組二：模型訓練與 genre 彙總
#
# 目標：
# 2.1 訓練偏好模型（Matrix Factorization / SVD + bias）
#     - 從 ratings 學出 U / V（以及可選的 bias 與 global mean）
#     - 並將模型參數存成 artifacts 供線上推論使用
#
# 2.2 Genre 彙總器（Genre Aggregation）
#     - 將「對每部電影的預測分數」轉成「對每個 genre 的分數」
#     - 典型策略：Top-M pooling（每個 genre 取分數最高的 M 部電影平均）
#
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 共同：讀檔工具（承接模組一輸出的 artifacts）
# ============================================================

def _load_df(path_no_ext: str) -> pd.DataFrame:
    """
    依副檔名自動載入 parquet / csv。
    你模組一支援兩者輸出，這裡做相容。
    """
    if os.path.exists(path_no_ext + ".parquet"):
        return pd.read_parquet(path_no_ext + ".parquet")
    if os.path.exists(path_no_ext + ".csv"):
        return pd.read_csv(path_no_ext + ".csv")
    raise FileNotFoundError(f"找不到資料檔：{path_no_ext}.parquet 或 {path_no_ext}.csv")


def load_preprocess_artifacts(preprocess_dir: str):
    """
    載入模組一的輸出（artifacts/preprocess/）
    預期包含：
      - train.(parquet|csv), test.(parquet|csv)
      - movies.(parquet|csv)  (內含 movie_idx 與 genre multi-hot 欄位)
      - encoders.json, meta.json
    """
    train_df = _load_df(os.path.join(preprocess_dir, "train"))
    test_df  = _load_df(os.path.join(preprocess_dir, "test"))
    movies_df = _load_df(os.path.join(preprocess_dir, "movies"))

    with open(os.path.join(preprocess_dir, "encoders.json"), "r", encoding="utf-8") as f:
        encoders = json.load(f)
    with open(os.path.join(preprocess_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return train_df, test_df, movies_df, encoders, meta


# ============================================================
# 2.1 偏好模型（Matrix Factorization / SVD + biases）
# ============================================================

@dataclass
class MFConfig:
    """
    Matrix Factorization 訓練超參數設定。
    """
    latent_dim: int = 64          # embedding 維度 d
    epochs: int = 20              # 訓練次數
    lr: float = 0.01              # learning rate
    reg: float = 0.02             # L2 正則化係數（同時作用於 U / V / bias）
    use_bias: bool = True         # 是否使用 user/movie bias + global mean
    seed: int = 42                # 隨機種子
    shuffle_each_epoch: bool = True

    # 訓練穩定性：避免極端更新（可選）
    clip_grad: Optional[float] = None  # e.g. 5.0

    # 顯示/評估頻率
    eval_every: int = 1


@dataclass
class MFArtifacts:
    """
    模型訓練後要落盤的 artifacts。
    線上推論時，你只需要載入這些即可做打分：
      score(u,i,j) = global_mean + bu[i] + bv[j] + U[i]·V[j]
    """
    U: np.ndarray                 # [num_users, d]
    V: np.ndarray                 # [num_movies, d]
    bias_user: Optional[np.ndarray]  # [num_users]
    bias_movie: Optional[np.ndarray] # [num_movies]
    global_mean: float
    config: MFConfig

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "U.npy"), self.U)
        np.save(os.path.join(out_dir, "V.npy"), self.V)

        # bias 可能關閉
        if self.bias_user is not None:
            np.save(os.path.join(out_dir, "bias_user.npy"), self.bias_user)
        if self.bias_movie is not None:
            np.save(os.path.join(out_dir, "bias_movie.npy"), self.bias_movie)

        with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir, "global_mean.json"), "w", encoding="utf-8") as f:
            json.dump({"global_mean": float(self.global_mean)}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(model_dir: str) -> "MFArtifacts":
        U = np.load(os.path.join(model_dir, "U.npy"))
        V = np.load(os.path.join(model_dir, "V.npy"))

        bias_user_path = os.path.join(model_dir, "bias_user.npy")
        bias_movie_path = os.path.join(model_dir, "bias_movie.npy")

        bias_user = np.load(bias_user_path) if os.path.exists(bias_user_path) else None
        bias_movie = np.load(bias_movie_path) if os.path.exists(bias_movie_path) else None

        with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
            cfg = MFConfig(**json.load(f))

        with open(os.path.join(model_dir, "global_mean.json"), "r", encoding="utf-8") as f:
            global_mean = float(json.load(f)["global_mean"])

        return MFArtifacts(
            U=U, V=V, bias_user=bias_user, bias_movie=bias_movie,
            global_mean=global_mean, config=cfg
        )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mf_predict_batch(
    user_idx: np.ndarray,
    movie_idx: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    global_mean: float,
    bias_user: Optional[np.ndarray],
    bias_movie: Optional[np.ndarray],
) -> np.ndarray:
    """
    向量化預測：對一批 (user_idx, movie_idx) 產生預測分數。
    """
    pred = np.sum(U[user_idx] * V[movie_idx], axis=1) + global_mean
    if bias_user is not None:
        pred += bias_user[user_idx]
    if bias_movie is not None:
        pred += bias_movie[movie_idx]
    return pred


def train_mf_sgd(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_users: int,
    num_movies: int,
    cfg: MFConfig
) -> MFArtifacts:
    """
    使用 SGD 訓練 explicit-feedback 的 Matrix Factorization（含 bias）。
    目標：最小化
      (r_ui - (mu + bu[u] + bv[i] + U[u]·V[i]))^2 + reg*(||U||^2 + ||V||^2 + ||bu||^2 + ||bv||^2)

    注意：
    - 這是最務實可控、容易 debug 的 baseline。
    - 後續你若要換 implicit（如 BPR/ALS），也可沿用 artifacts 介面（U/V/bias）。
    """
    rng = np.random.default_rng(cfg.seed)

    # 1) 取訓練資料為 numpy（加速迴圈）
    u_train = train_df["user_idx"].to_numpy(dtype=np.int64)
    m_train = train_df["movie_idx"].to_numpy(dtype=np.int64)
    r_train = train_df["rating"].to_numpy(dtype=np.float32)

    u_test = test_df["user_idx"].to_numpy(dtype=np.int64)
    m_test = test_df["movie_idx"].to_numpy(dtype=np.int64)
    r_test = test_df["rating"].to_numpy(dtype=np.float32)

    # 2) 初始化參數
    # 常見做法：U/V 用小常態分佈；bias 初始化為 0
    U = 0.01 * rng.standard_normal((num_users, cfg.latent_dim), dtype=np.float32)
    V = 0.01 * rng.standard_normal((num_movies, cfg.latent_dim), dtype=np.float32)

    global_mean = float(np.mean(r_train))

    bias_user = np.zeros(num_users, dtype=np.float32) if cfg.use_bias else None
    bias_movie = np.zeros(num_movies, dtype=np.float32) if cfg.use_bias else None

    # 3) 訓練主迴圈
    idx = np.arange(len(r_train), dtype=np.int64)

    for epoch in range(1, cfg.epochs + 1):
        if cfg.shuffle_each_epoch:
            rng.shuffle(idx)

        # SGD：逐筆更新
        for t in idx:
            u = u_train[t]
            i = m_train[t]
            r = r_train[t]

            # 預測
            pred = float(np.dot(U[u], V[i]) + global_mean)
            if cfg.use_bias:
                pred += float(bias_user[u]) + float(bias_movie[i])

            err = r - pred  # 梯度方向：讓 pred 更接近 r

            # 可選：梯度裁切（避免爆炸）
            if cfg.clip_grad is not None:
                err = float(np.clip(err, -cfg.clip_grad, cfg.clip_grad))

            # 更新 bias（若使用）
            if cfg.use_bias:
                bu = bias_user[u]
                bi = bias_movie[i]

                bias_user[u] = bu + cfg.lr * (err - cfg.reg * bu)
                bias_movie[i] = bi + cfg.lr * (err - cfg.reg * bi)

            # 更新 U/V（同時做 L2 正則）
            # d/dU: -2*err*V + 2*reg*U
            # d/dV: -2*err*U + 2*reg*V
            Uu = U[u].copy()
            Vi = V[i].copy()

            U[u] = Uu + cfg.lr * (err * Vi - cfg.reg * Uu)
            V[i] = Vi + cfg.lr * (err * Uu - cfg.reg * Vi)

        # 4) 評估（每 eval_every epoch）
        if (epoch % cfg.eval_every) == 0:
            pred_train = mf_predict_batch(u_train, m_train, U, V, global_mean, bias_user, bias_movie)
            pred_test  = mf_predict_batch(u_test,  m_test,  U, V, global_mean, bias_user, bias_movie)

            train_rmse = rmse(r_train, pred_train)
            test_rmse  = rmse(r_test,  pred_test)

            print(f"[MF][Epoch {epoch:03d}] train_rmse={train_rmse:.4f}  test_rmse={test_rmse:.4f}")

    return MFArtifacts(
        U=U,
        V=V,
        bias_user=bias_user,
        bias_movie=bias_movie,
        global_mean=global_mean,
        config=cfg
    )


# ============================================================
# 2.2 Genre 彙總器（Top-M pooling）
# ============================================================

@dataclass
class GenreAggConfig:
    """
    將電影分數彙總成 genre 分數的策略設定。
    """
    top_m_pool: int = 50   # 每個 genre 取分數最高的 M 部電影做平均（Top-M pooling）
    top_k_genres: int = 5  # 最終輸出 Top-K genres
    top_n_movies_per_genre: int = 5  # 每個 genre 額外輸出代表電影 Top-N（可做可視化/解釋）


class GenreAggregator:
    """
    Genre 彙總器：輸入 user 向量 u，輸出 genre 分數與 Top-K genre。
    這層是「產品輸出層」，因為 MF 原生輸出是「電影分數」不是「類型分數」。
    """

    def __init__(
        self,
        V: np.ndarray,
        movies_df: pd.DataFrame,
        genre_vocab: Dict[str, int],
        bias_movie: Optional[np.ndarray] = None,
        global_mean: float = 0.0,
    ):
        """
        V: [num_movies, d]
        movies_df: 必須包含 movie_idx 與 multi-hot 欄位 genre_*
        genre_vocab: {genre_name: col_index}
        """
        self.V = V.astype(np.float32)
        self.bias_movie = bias_movie.astype(np.float32) if bias_movie is not None else None
        self.global_mean = float(global_mean)

        # 以 movie_idx 對齊 V 的列：非常重要
        # movies_df 在模組一已保證只保留有互動的電影，且有 movie_idx
        if "movie_idx" not in movies_df.columns:
            raise ValueError("movies_df 必須包含 movie_idx 欄位（模組一已產生）")

        self.movies_df = movies_df.copy()
        self.genre_vocab = dict(genre_vocab)

        # multi-hot 欄位名稱慣例：genre_{GenreName}
        # 注意 GenreName 可能含特殊字元，這裡以模組一產生的欄位為準
        self.genre_cols = []
        for g in self.genre_vocab.keys():
            col = f"genre_{g}"
            if col not in self.movies_df.columns:
                raise ValueError(f"movies_df 找不到 genre multi-hot 欄位：{col}")
            self.genre_cols.append(col)

        # 建立每個 genre -> movie_idx list，加速彙總
        self.genre_to_movie_idxs: Dict[str, np.ndarray] = {}
        for g, col in zip(self.genre_vocab.keys(), self.genre_cols):
            # 取屬於該 genre 的電影 movie_idx
            mids = self.movies_df.loc[self.movies_df[col] == 1, "movie_idx"].to_numpy(dtype=np.int64)
            self.genre_to_movie_idxs[g] = mids

    def score_movies(self, u: np.ndarray) -> np.ndarray:
        """
        對所有電影打分：s_j = u · V[j] (+ global_mean + bias_movie[j])
        回傳 shape = [num_movies]
        """
        u = u.astype(np.float32)
        scores = self.V @ u  # [num_movies]
        scores = scores + self.global_mean
        if self.bias_movie is not None:
            scores = scores + self.bias_movie
        return scores

    def aggregate(self, u: np.ndarray, top_m_pool: int, top_k_genres: int, top_n_movies_per_genre: int):
        """
        產出：
          - genre_scores: Dict[genre, score]
          - top_genres: List[Tuple[genre, score]]
          - genre_top_movies: Dict[genre, List[Tuple[movie_idx, title, score]]]
        """
        movie_scores = self.score_movies(u)

        genre_scores: Dict[str, float] = {}
        genre_top_movies: Dict[str, List[Tuple[int, str, float]]] = {}

        for g, mids in self.genre_to_movie_idxs.items():
            if len(mids) == 0:
                genre_scores[g] = float("-inf")
                genre_top_movies[g] = []
                continue

            # 取該 genre 的電影分數
            s = movie_scores[mids]

            # Top-M pooling：取分數最高的 M 部電影平均（若不足 M，就全取）
            M = min(top_m_pool, len(s))
            top_idx_local = np.argpartition(-s, M - 1)[:M]  # O(n) 選 top M（不完全排序）
            top_scores = s[top_idx_local]
            genre_scores[g] = float(np.mean(top_scores))

            # 額外輸出代表電影 Top-N（做完整排序，N 很小）
            N = min(top_n_movies_per_genre, len(s))
            topN_local = np.argsort(-s)[:N]
            top_mids = mids[topN_local]
            top_mscores = s[topN_local]

            # movie_idx -> title（若 title 欄位存在）
            # 模組一 movies_df 內有 title 欄位（你 rename 過）
            rows = self.movies_df.set_index("movie_idx").loc[top_mids]
            titles = rows["title"].to_list() if "title" in rows.columns else [""] * len(top_mids)

            genre_top_movies[g] = [
                (int(mi), str(ti), float(sc))
                for mi, ti, sc in zip(top_mids, titles, top_mscores)
            ]

        # 取 Top-K genres
        top_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_genres]

        return genre_scores, top_genres, genre_top_movies


# ============================================================
# 模組二 Pipeline：訓練 + 存檔 + demo（genre 彙總）
# ============================================================

def train_and_save(
    preprocess_dir: str,
    model_out_dir: str,
    mf_cfg: MFConfig,
) -> MFArtifacts:
    """
    訓練的主程式：
    - 載入模組一 artifacts
    - 訓練 MF
    - 存下 U/V/bias/config
    """
    train_df, test_df, movies_df, encoders, meta = load_preprocess_artifacts(preprocess_dir)

    num_users = int(meta["n_users"])
    num_movies = int(meta["n_movies"])

    # 基本欄位檢查（避免 artifacts 不一致）
    required_cols = {"user_idx", "movie_idx", "rating"}
    if not required_cols.issubset(set(train_df.columns)):
        raise ValueError(f"train_df 缺少必要欄位：{required_cols - set(train_df.columns)}")
    if not required_cols.issubset(set(test_df.columns)):
        raise ValueError(f"test_df 缺少必要欄位：{required_cols - set(test_df.columns)}")

    artifacts = train_mf_sgd(
        train_df=train_df,
        test_df=test_df,
        num_users=num_users,
        num_movies=num_movies,
        cfg=mf_cfg
    )

    artifacts.save(model_out_dir)
    print(f"模型已存至：{model_out_dir}")
    return artifacts


def demo_genre_aggregation(
    preprocess_dir: str,
    model_dir: str,
    user_idx: int,
    agg_cfg: GenreAggConfig
):
    """
    示範：載入訓練好的 MF artifacts，對指定 user_idx 輸出 Top-K genres。
    """
    _, _, movies_df, encoders, _ = load_preprocess_artifacts(preprocess_dir)
    mf = MFArtifacts.load(model_dir)

    genre_vocab = encoders["genre_vocab"]  # {genre: idx}
    aggregator = GenreAggregator(
        V=mf.V,
        movies_df=movies_df,
        genre_vocab=genre_vocab,
        bias_movie=mf.bias_movie,
        global_mean=mf.global_mean
    )

    u = mf.U[user_idx]
    genre_scores, top_genres, genre_top_movies = aggregator.aggregate(
        u=u,
        top_m_pool=agg_cfg.top_m_pool,
        top_k_genres=agg_cfg.top_k_genres,
        top_n_movies_per_genre=agg_cfg.top_n_movies_per_genre
    )

    print(f"\n[User {user_idx}] Top-{agg_cfg.top_k_genres} genres:")
    for g, s in top_genres:
        print(f"  - {g:15s} score={s:.4f}")

    print("\n代表電影（每個 Top genre 列出前 N 部）:")
    for g, _ in top_genres:
        print(f"\n[{g}]")
        for mi, title, sc in genre_top_movies[g]:
            print(f"  movie_idx={mi:5d}  score={sc:.4f}  title={title}")


# ============================================================
# 主程式入口（可直接跑）
# ============================================================

if __name__ == "__main__":
    preprocess_dir = "artifacts/preprocess"
    model_out_dir = "artifacts/mf_model"

    # 你可以先用小一點的維度/epoch 確認流程跑通，再加大
    mf_cfg = MFConfig(
        latent_dim=64,
        epochs=20,
        lr=0.01,
        reg=0.02,
        use_bias=True,
        seed=42,
        clip_grad=5.0,
        eval_every=1,
    )

    # 1) 訓練並存檔
    train_and_save(
        preprocess_dir=preprocess_dir,
        model_out_dir=model_out_dir,
        mf_cfg=mf_cfg
    )

    # 2) demo：對某個 user_idx 做 genre 彙總
    agg_cfg = GenreAggConfig(top_m_pool=50, top_k_genres=5, top_n_movies_per_genre=5)
    demo_genre_aggregation(
        preprocess_dir=preprocess_dir,
        model_dir=model_out_dir,
        user_idx=0,
        agg_cfg=agg_cfg
    )
