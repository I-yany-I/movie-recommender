"""基于BERT的文本内容推荐模型 — 内存友好版本"""
import numpy as np
import pandas as pd

from .base import BaseRecommender
import logging

logger = logging.getLogger(__name__)


class ContentBERTModel(BaseRecommender):
    """利用预计算BERT嵌入做基于内容的推荐"""

    def __init__(self):
        super().__init__(name="content_bert")
        self.movie_embeddings: np.ndarray = None  # (n_items, dim)
        self.movie_ids: np.ndarray = None          # 嵌入对应的movie_idx
        self.idx_to_pos: dict = {}                 # movie_idx → position in embeddings

    def train(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None):
        """Load pre-computed BERT embeddings and build lookup tables."""
        self.load_data(train_df, val_df)

        emb_dir = self.config.get_project_root() / "data/processed/embeddings"
        emb_path = emb_dir / "movie_bert_embeddings.npy"
        ids_path = emb_dir / "movie_bert_ids.npy"

        if not emb_path.exists():
            raise FileNotFoundError(
                f"BERT嵌入未找到: {emb_path}。请先运行 DataAgent 进行特征工程。"
            )

        self.movie_embeddings = np.load(emb_path)
        if ids_path.exists():
            self.movie_ids = np.load(ids_path)
        else:
            self.movie_ids = np.arange(len(self.movie_embeddings))

        self.idx_to_pos = {int(mid): i for i, mid in enumerate(self.movie_ids)}

        logger.info(f"[ContentBERT] 加载嵌入: {self.movie_embeddings.shape}, "
                     f"{len(self.idx_to_pos)} 电影索引")

    def _get_similarity_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """计算query嵌入与所有电影的余弦相似度（嵌入已L2归一化）"""
        return self.movie_embeddings @ query_embedding

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Global mean as fallback prediction — content model not suited for rating prediction."""
        # ContentBERT is a content-similarity model, not a rating predictor.
        # Return the global mean rating for compatibility with the evaluation framework.
        if self.train_data is not None:
            global_mean = float(self.train_data["rating"].mean())
        else:
            global_mean = 3.5
        return np.full(len(user_ids), global_mean)

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> list:
        """基于用户最近观影历史的嵌入均值找最相似电影"""
        seen = set()
        user_vecs = []

        if self.train_data is not None:
            # Filter efficiently: use .loc with pre-sorted data if available
            user_mask = self.train_data["user_idx"] == user_idx
            user_rows = self.train_data.loc[user_mask]
            seen = set(user_rows["movie_idx"].values)

            # Build user profile from rated movies
            for mid in user_rows["movie_idx"].values:
                if int(mid) in self.idx_to_pos:
                    user_vecs.append(self.movie_embeddings[self.idx_to_pos[int(mid)]])

        if not user_vecs:
            # Cold start: return most popular / highest average rated movies
            # Use movie embeddings centroid of all movies as fallback
            user_vec = self.movie_embeddings.mean(axis=0)
        else:
            user_vec = np.mean(user_vecs, axis=0)

        # Exclude seen items
        valid_mask = np.ones(len(self.movie_embeddings), dtype=bool)
        for s in seen:
            if int(s) in self.idx_to_pos:
                valid_mask[self.idx_to_pos[int(s)]] = False

        if valid_mask.sum() == 0:
            return []

        similarities = self.movie_embeddings[valid_mask] @ user_vec
        valid_indices = np.where(valid_mask)[0]
        top_local = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in top_local:
            mid = int(self.movie_ids[valid_indices[i]])
            results.append((mid, float(similarities[i])))
        return results
