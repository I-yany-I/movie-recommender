"""推荐模型抽象基类"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import Config

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """所有推荐模型的抽象基类"""

    def __init__(self, name: str):
        self.name = name
        self.config = Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = self.config.get_project_root() / "data/processed/models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 数据和元信息
        self.n_users: int = 0
        self.n_items: int = 0
        self.train_data: pd.DataFrame = None
        self.val_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """训练模型"""
        ...

    @abstractmethod
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """预测单个用户-物品对的评分"""
        ...

    @abstractmethod
    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> list:
        """为指定用户生成推荐列表 [(item_idx, score), ...]"""
        ...

    def load_data(self, train_df: pd.DataFrame = None,
                  val_df: pd.DataFrame = None, test_df: pd.DataFrame = None):
        """Load train/val/test data and compute user/item dimensions."""
        processed_dir = self.config.get_project_root() / "data/processed"

        if train_df is None:
            train_df = pd.read_parquet(processed_dir / "ratings.parquet")
        # Avoid full sort on 20M+ rows — sort only when a model needs it via _ensure_sorted()
        self.train_data = train_df
        self._train_sorted = False

        if val_df is not None:
            self.val_data = val_df
        if test_df is not None:
            self.test_data = test_df

        # Calculate dimensions from all available data without concatenating
        max_user = int(self.train_data["user_idx"].max())
        max_item = int(self.train_data["movie_idx"].max())
        if self.val_data is not None:
            max_user = max(max_user, int(self.val_data["user_idx"].max()))
            max_item = max(max_item, int(self.val_data["movie_idx"].max()))
        if self.test_data is not None:
            max_user = max(max_user, int(self.test_data["user_idx"].max()))
            max_item = max(max_item, int(self.test_data["movie_idx"].max()))
        self.n_users = max_user + 1
        self.n_items = max_item + 1

        logger.info(f"[{self.name}] Data: {self.n_users} users x {self.n_items} items")

    def evaluate(self, test_df: pd.DataFrame = None) -> Dict[str, float]:
        """在测试集上评估模型"""
        from ..evaluation.metrics import (
            compute_rmse,
            compute_mae,
            build_test_dict,
            evaluate_recommendations,
        )

        if test_df is not None:
            self.test_data = test_df
        if self.test_data is None:
            raise RuntimeError("无测试数据，请先调用 load_data()")

        # 评分预测评估
        y_true = self.test_data["rating"].values
        user_ids = self.test_data["user_idx"].values
        item_ids = self.test_data["movie_idx"].values
        y_pred = self.predict(user_ids, item_ids)

        rmse_val = compute_rmse(y_true, y_pred)
        mae_val = compute_mae(y_true, y_pred)

        # 排序评估 — 采样用户以加速（全量162K用户太慢）
        test_dict = build_test_dict(self.test_data)
        predictions = {}
        sampled_users = list(test_dict.keys())
        if len(sampled_users) > 500:
            rng = np.random.RandomState(42)
            sampled_users = rng.choice(sampled_users, 500, replace=False).tolist()
        for uid in sampled_users:
            predictions[uid] = self.recommend(uid, top_k=10)
        ranking_metrics = evaluate_recommendations(predictions, test_dict, k=10)

        self.metrics = {
            "rmse": rmse_val,
            "mae": mae_val,
            **ranking_metrics,
        }

        logger.info(f"[{self.name}] 评估结果: {self.metrics}")
        return self.metrics

    def save(self):
        """保存模型"""
        import joblib
        path = self.model_dir / f"{self.name}_model.pkl"
        joblib.dump(self, path)
        logger.info(f"[{self.name}] 模型已保存: {path}")

    @classmethod
    def load(cls, name: str):
        """加载模型"""
        import joblib
        model_dir = Config.get_project_root() / "data/processed/models"
        path = model_dir / f"{name}_model.pkl"
        if path.exists():
            return joblib.load(path)
        raise FileNotFoundError(f"模型文件不存在: {path}")
