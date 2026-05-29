"""SVD++ 矩阵分解推荐模型 (基于Surprise库)"""
import numpy as np
import pandas as pd
from surprise import SVDpp, Reader, Dataset
from surprise.model_selection import train_test_split as surprise_split

from .base import BaseRecommender
import logging

logger = logging.getLogger(__name__)


class SVDModel(BaseRecommender):
    """SVD++ 协同过滤"""

    def __init__(self):
        super().__init__(name="svd")
        cfg = self.config.get("models.svd", {})
        self.n_factors = cfg.get("n_factors", 100)
        self.n_epochs = cfg.get("n_epochs", 20)
        self.lr = cfg.get("lr_all", 0.005)
        self.reg = cfg.get("reg_all", 0.02)
        self.model = None
        self.global_mean: float = 0.0

    def train(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None):
        self.load_data(train_df, val_df)
        self.global_mean = float(self.train_data["rating"].mean())

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.train_data[["user_idx", "movie_idx", "rating"]],
            reader,
        )
        trainset = data.build_full_trainset()

        logger.info(f"[SVD] 开始训练 (k={self.n_factors}, epochs={self.n_epochs})")
        self.model = SVDpp(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr,
            reg_all=self.reg,
            verbose=True,
        )
        self.model.fit(trainset)
        logger.info("[SVD] 训练完成")

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        preds = np.array([
            self.model.predict(int(uid), int(iid)).est
            for uid, iid in zip(user_ids, item_ids)
        ])
        return preds

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> list:
        seen = set()
        if exclude_seen and self.train_data is not None:
            user_data = self.train_data[self.train_data["user_idx"] == user_idx]
            seen = set(user_data["movie_idx"].values)

        candidates = [i for i in range(self.n_items) if i not in seen]
        if not candidates:
            return []

        uid = int(user_idx)
        # Batch predict via Surprise test() for speed
        testset = [(uid, iid, 0) for iid in candidates]
        predictions = self.model.test(testset)
        scores = [(int(p.iid), p.est) for p in predictions]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
