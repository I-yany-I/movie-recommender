"""热门推荐基线模型"""
import numpy as np
import pandas as pd

from .base import BaseRecommender
import logging

logger = logging.getLogger(__name__)


class PopularityModel(BaseRecommender):
    """基于电影加权评分的非个性化热门推荐"""

    def __init__(self):
        super().__init__(name="popularity")
        self.popularity_scores: np.ndarray = None  # shape: (n_items,)

    def train(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None):
        """计算每部电影的加权评分"""
        self.load_data(train_df, val_df)

        # 贝叶斯加权：WR = (v/(v+m))*R + (m/(v+m))*C
        # R = 电影平均分, v = 电影评分数, C = 全局平均分, m = 最小评分阈值
        movie_stats = self.train_data.groupby("movie_idx").agg(
            mean_rating=("rating", "mean"),
            count=("rating", "count"),
        )

        C = float(self.train_data["rating"].mean())
        m = float(movie_stats["count"].quantile(0.90))  # 取90分位数作为阈值

        scores = np.zeros(self.n_items)
        for idx, row in movie_stats.iterrows():
            v = row["count"]
            R = row["mean_rating"]
            scores[int(idx)] = (v / (v + m)) * R + (m / (v + m)) * C

        self.popularity_scores = scores
        logger.info(f"[Popularity] 训练完成，{len(movie_stats)}部电影")

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """返回电影的热门分数（已归一化到1-5范围）"""
        raw = self.popularity_scores[item_ids.astype(int)]
        # 归一化到1-5
        min_s, max_s = raw.min(), raw.max()
        if max_s > min_s:
            return 1 + 4 * (raw - min_s) / (max_s - min_s)
        return raw

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> list:
        """返回全局热门电影"""
        seen = set()
        if exclude_seen and self.train_data is not None:
            user_data = self.train_data[self.train_data["user_idx"] == user_idx]
            seen = set(user_data["movie_idx"].values)

        scores = self.popularity_scores.copy()
        for s in seen:
            if s < len(scores):
                scores[int(s)] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[int(i)])) for i in top_indices]
