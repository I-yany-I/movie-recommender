"""集成推荐模型 — 加权融合多个模型的推荐结果"""
import logging
from typing import Dict, List

import numpy as np

from .base import BaseRecommender

logger = logging.getLogger(__name__)


class EnsembleModel(BaseRecommender):
    """加权融合多模型推荐"""

    def __init__(self, models: Dict[str, BaseRecommender] = None,
                 weights: Dict[str, float] = None):
        super().__init__(name="ensemble")
        self.models = models or {}
        self.weights = weights or {}
        self._normalize_weights()

    def _normalize_weights(self):
        if not self.weights:
            self.weights = {name: 1.0 / len(self.models) for name in self.models}
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def add_model(self, model: BaseRecommender, weight: float = None):
        self.models[model.name] = model
        if weight is not None:
            self.weights[model.name] = weight
        else:
            self.weights[model.name] = 1.0
        self._normalize_weights()
        logger.info(f"[Ensemble] 添加模型 {model.name}，当前权重: {self.weights}")

    def train(self, train_df=None, val_df=None):
        if val_df is not None and len(self.models) > 1:
            self._optimize_weights(val_df)

    def _optimize_weights(self, val_df):
        """Random search for optimal weights using NDCG@10 on validation set."""
        from ..evaluation.metrics import build_test_dict, evaluate_recommendations

        model_names = list(self.models.keys())
        n_models = len(model_names)

        test_dict = build_test_dict(val_df)
        sampled_users = list(test_dict.keys())
        if len(sampled_users) > 200:
            rng = np.random.RandomState(42)
            sampled_users = rng.choice(sampled_users, 200, replace=False).tolist()
            test_dict = {u: test_dict[u] for u in sampled_users}

        best_weights = dict(zip(model_names, [1.0 / n_models] * n_models))
        best_ndcg = 0.0

        rng = np.random.RandomState(42)
        for _ in range(300):
            raw = rng.dirichlet(np.ones(n_models) * 2.0)
            candidate = dict(zip(model_names, raw))

            # Build weighted predictions for sampled users
            predictions = {}
            for uid in sampled_users:
                item_scores = {}
                for name, model in self.models.items():
                    w = candidate[name]
                    try:
                        recs = model.recommend(int(uid), top_k=30, exclude_seen=True)
                    except Exception:
                        continue
                    if not recs:
                        continue
                    scores_arr = np.array([s for _, s in recs])
                    s_min, s_max = scores_arr.min(), scores_arr.max()
                    if s_max > s_min:
                        scores_arr = (scores_arr - s_min) / (s_max - s_min)
                    else:
                        scores_arr = np.ones_like(scores_arr) * 0.5
                    for (iid, _), ns in zip(recs, scores_arr):
                        item_scores[iid] = item_scores.get(iid, 0.0) + w * ns

                sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                predictions[uid] = sorted_items

            metrics = evaluate_recommendations(predictions, test_dict, k=10)
            ndcg = metrics.get("ndcg@10", 0.0)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_weights = candidate

        self.weights = best_weights
        self._normalize_weights()
        logger.info(f"[Ensemble] 权重优化完成 (NDCG@10={best_ndcg:.4f}): {self.weights}")

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("集成模型中没有子模型")

        preds = np.zeros(len(user_ids))
        for name, model in self.models.items():
            w = self.weights.get(name, 1.0 / len(self.models))
            try:
                model_preds = model.predict(user_ids, item_ids)
                preds += w * model_preds
            except Exception as e:
                logger.warning(f"[Ensemble] 模型 {name} 预测失败: {e}")
        return preds

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> list:
        if not self.models:
            return []

        item_scores = {}
        for name, model in self.models.items():
            w = self.weights.get(name, 1.0 / len(self.models))
            try:
                recs = model.recommend(int(user_idx), top_k=50, exclude_seen=exclude_seen)
            except Exception as e:
                logger.warning(f"[Ensemble] 模型 {name} 推荐失败: {e}")
                continue
            if not recs:
                continue

            # Per-model min-max normalization before weighted fusion
            scores_arr = np.array([s for _, s in recs])
            s_min, s_max = scores_arr.min(), scores_arr.max()
            if s_max > s_min:
                scores_norm = (scores_arr - s_min) / (s_max - s_min)
            else:
                scores_norm = np.ones_like(scores_arr) * 0.5

            for (iid, _), ns in zip(recs, scores_norm):
                item_scores[iid] = item_scores.get(iid, 0.0) + w * ns

        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

    def load_models_from_names(self, model_names: List[str]):
        for name in model_names:
            try:
                model = BaseRecommender.load(name)
                self.add_model(model)
            except FileNotFoundError:
                logger.warning(f"[Ensemble] 模型 {name} 未找到，跳过")
