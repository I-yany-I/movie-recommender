"""数据预处理模块 — 清洗、过滤、编码"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from ..config import Config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器 — 加载原始CSV/TSV并进行清洗过滤"""

    def __init__(self):
        self.config = Config
        self.project_root = Config.get_project_root()
        self.min_user = Config.get("preprocessing.min_user_ratings", 20)
        self.min_movie = Config.get("preprocessing.min_movie_ratings", 20)
        self.seed = Config.get("preprocessing.random_seed", 42)

        # 编码器
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

        # 数据容器
        self.ratings_df: pd.DataFrame = None
        self.movies_df: pd.DataFrame = None
        self.tags_df: pd.DataFrame = None

        # 处理后的数据
        self.filtered_ratings: pd.DataFrame = None
        self.n_users: int = 0
        self.n_movies: int = 0
        self.n_ratings: int = 0
        self.sparsity: float = 0.0

    def run(self) -> dict:
        """执行预处理流水线"""
        logger.info("=" * 50)
        logger.info("开始数据预处理...")
        logger.info("=" * 50)

        self._load_raw_data()
        self._filter_and_encode()
        stats = self._compute_statistics()
        self._save_processed()
        logger.info(f"预处理完成: {self.n_users}用户, {self.n_movies}电影, {self.n_ratings}评分")
        return stats

    def _load_raw_data(self):
        """加载MovieLens原始数据"""
        ml_path = self.project_root / Config.get("data.movielens.local_path")
        logger.info(f"加载 MovieLens 数据从 {ml_path}")

        self.ratings_df = pd.read_csv(ml_path / "ratings.csv")
        self.movies_df = pd.read_csv(ml_path / "movies.csv")
        tags_path = ml_path / "tags.csv"
        if tags_path.exists():
            self.tags_df = pd.read_csv(tags_path)

        logger.info(
            f"原始数据: {self.ratings_df.shape[0]}条评分, "
            f"{self.movies_df.shape[0]}部电影, "
            f"{self.ratings_df['userId'].nunique()}用户"
        )

    def _filter_and_encode(self):
        """过滤冷门用户/电影，并重新编码ID"""
        # 统计用户和电影的评分次数
        user_counts = self.ratings_df["userId"].value_counts()
        movie_counts = self.ratings_df["movieId"].value_counts()

        valid_users = user_counts[user_counts >= self.min_user].index
        valid_movies = movie_counts[movie_counts >= self.min_movie].index

        logger.info(
            f"过滤后: {len(valid_users)}用户(>={self.min_user}评分), "
            f"{len(valid_movies)}电影(>={self.min_movie}评分)"
        )

        # 过滤
        mask = (
            self.ratings_df["userId"].isin(valid_users)
            & self.ratings_df["movieId"].isin(valid_movies)
        )
        self.filtered_ratings = self.ratings_df[mask].copy()

        # 重新编码为连续索引
        self.filtered_ratings["user_idx"] = self.user_encoder.fit_transform(
            self.filtered_ratings["userId"]
        )
        self.filtered_ratings["movie_idx"] = self.movie_encoder.fit_transform(
            self.filtered_ratings["movieId"]
        )

        self.n_users = len(self.user_encoder.classes_)
        self.n_movies = len(self.movie_encoder.classes_)
        self.n_ratings = len(self.filtered_ratings)

    def _compute_statistics(self) -> dict:
        """计算数据集统计信息"""
        total_cells = self.n_users * self.n_movies
        self.sparsity = 1.0 - (self.n_ratings / total_cells)

        stats = {
            "n_users": self.n_users,
            "n_movies": self.n_movies,
            "n_ratings": self.n_ratings,
            "sparsity": f"{self.sparsity:.4%}",
            "avg_ratings_per_user": f"{self.n_ratings / self.n_users:.1f}",
            "avg_ratings_per_movie": f"{self.n_ratings / self.n_movies:.1f}",
            "rating_mean": float(self.filtered_ratings["rating"].mean()),
            "rating_std": float(self.filtered_ratings["rating"].std()),
        }
        return stats

    def _save_processed(self):
        """保存处理后的数据为parquet格式"""
        out_dir = self.project_root / "data/processed"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.filtered_ratings.to_parquet(
            out_dir / "ratings.parquet", index=False
        )
        # 同时保存过滤后的电影表
        valid_movie_ids = set(self.filtered_ratings["movieId"].unique())
        filtered_movies = self.movies_df[
            self.movies_df["movieId"].isin(valid_movie_ids)
        ].copy()
        filtered_movies["movie_idx"] = self.movie_encoder.transform(
            filtered_movies["movieId"]
        )
        filtered_movies.to_parquet(out_dir / "movies.parquet", index=False)

        # 保存编码器映射
        import joblib
        joblib.dump(self.user_encoder, out_dir / "user_encoder.pkl")
        joblib.dump(self.movie_encoder, out_dir / "movie_encoder.pkl")

        logger.info(f"处理后数据已保存到 {out_dir}")

    def get_train_test_split(self) -> tuple:
        """Time-series split: each user's latest ratings → test/val, older → train."""
        if self.filtered_ratings is None:
            raise RuntimeError("请先调用 run() 方法")

        test_ratio = Config.get("preprocessing.test_split_ratio", 0.1)
        val_ratio = Config.get("preprocessing.val_split_ratio", 0.1)

        # Check cached splits first
        out_dir = Config.get_project_root() / "data/processed"
        cache_train = out_dir / "train.parquet"
        cache_test = out_dir / "test.parquet"
        cache_val = out_dir / "val.parquet"
        if cache_train.exists() and cache_test.exists():
            logger.info("从缓存加载训练/验证/测试划分")
            return (
                pd.read_parquet(cache_train),
                pd.read_parquet(cache_val),
                pd.read_parquet(cache_test),
            )

        df = self.filtered_ratings.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

        train_parts, val_parts, test_parts = [], [], []
        for _, user_data in df.groupby("user_idx", sort=False):
            n = len(user_data)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = max(1, n - n_test)
                n_val = 0
            train_parts.append(user_data.iloc[:n_train])
            if n_val > 0:
                val_parts.append(user_data.iloc[n_train:n_train + n_val])
            test_parts.append(user_data.iloc[n_train + n_val:])

        train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=self.seed)
        val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=self.seed) if val_parts else pd.DataFrame()
        test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=self.seed)

        logger.info(
            f"Time-series split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}"
        )

        # Cache splits
        train_df.to_parquet(cache_train, index=False)
        val_df.to_parquet(cache_val, index=False)
        test_df.to_parquet(cache_test, index=False)

        return train_df, val_df, test_df
