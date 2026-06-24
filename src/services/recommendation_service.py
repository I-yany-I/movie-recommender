"""推荐服务 — 统一推荐接口，包装所有模型和Agent"""
import logging
from typing import Dict, List, Optional

import numpy as np

from .movie_service import MovieService

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    推荐服务 — 提供统一的推荐接口，可以调度多个模型。

    当前以基于规则的推荐为主（通过MovieService），
    模型预测接口预留给后续扩展。
    """

    def __init__(self, movie_service: MovieService = None):
        self.movies = movie_service
        self._models = {}  # 延迟加载的模型
        self._ensemble_model = None

    def load_models(self) -> Dict[str, object]:
        """加载所有已训练的推荐模型"""
        from pathlib import Path
        from ..config import Config
        import pickle

        model_dir = Config.get_project_root() / "data/processed/models"
        model_names = ["popularity", "svd", "ncf", "lightgcn", "content_bert", "ensemble"]

        for name in model_names:
            model_path = model_dir / f"{name}_model.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        self._models[name] = pickle.load(f)
                    logger.info(f"已加载模型: {name}")
                except Exception as e:
                    logger.warning(f"模型加载失败 {name}: {e}")

        # ensemble特殊处理
        if "ensemble" in self._models:
            self._ensemble_model = self._models["ensemble"]

        return self._models

    def recommend_by_genres(
        self, genres: str, top_k: int = 20, exclude_seen: List[int] = None
    ) -> List[dict]:
        """按类型偏好推荐"""
        if self.movies:
            return self.movies.get_by_genre(genres, top_k)
        return []

    def recommend_similar(self, movie_idx: int, top_k: int = 8) -> List[dict]:
        """基于BERT的相似电影推荐"""
        if self.movies:
            return self.movies.get_similar(movie_idx, top_k)
        return []

    def recommend_popular(self, top_k: int = 20) -> List[dict]:
        """热门电影推荐"""
        if self.movies:
            return self.movies.get_popular(top_k)
        return []

    def recommend_with_models(
        self, user_id: int = None, top_k: int = 10
    ) -> List[dict]:
        """使用训练好的模型进行推荐（需要模型已加载）"""
        if not self._models:
            self.load_models()

        # 优先使用ensemble模型
        if self._ensemble_model and hasattr(self._ensemble_model, "recommend"):
            try:
                recs = self._ensemble_model.recommend(user_id or 42, top_k)
                if recs and self.movies:
                    movie_indices = [r[0] if isinstance(r, tuple) else r for r in recs]
                    return self.movies.get_by_ids(movie_indices)
            except Exception as e:
                logger.warning(f"Ensemble模型推荐失败: {e}")

        return self.recommend_popular(top_k)
