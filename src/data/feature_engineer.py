"""特征工程模块 — BERT文本向量化 + 知识图谱三元组构建"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..config import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程师 — 构建文本嵌入和知识图谱三元组"""

    def __init__(self, device: str = None):
        self.config = Config
        self.project_root = Config.get_project_root()
        self.processed_dir = self.project_root / "data/processed"
        self.embeddings_dir = self.processed_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"特征工程使用设备: {self.device}")

        self.bert_model = None
        self.movie_embeddings: np.ndarray = None

    def run(self) -> dict:
        """执行完整特征工程流水线"""
        logger.info("=" * 50)
        logger.info("开始特征工程...")
        logger.info("=" * 50)

        result = {}
        result["bert"] = self.build_bert_embeddings()
        result["kg"] = self.build_knowledge_graph_triples()
        result["genre_matrix"] = self.build_genre_features()
        return result

    def build_bert_embeddings(self) -> np.ndarray:
        """用BERT对电影概述进行编码"""
        model_name = Config.get("models.content_bert.model_name")
        batch_size = Config.get("models.content_bert.batch_size", 64)

        # 检查缓存
        cache_path = self.embeddings_dir / "movie_bert_embeddings.npy"
        if cache_path.exists():
            logger.info(f"从缓存加载BERT嵌入: {cache_path}")
            self.movie_embeddings = np.load(cache_path)
            return self.movie_embeddings

        logger.info(f"加载 BERT 模型: {model_name}")
        self.bert_model = SentenceTransformer(model_name, device=self.device)

        # 加载统一电影表
        unified_path = self.processed_dir / "unified_movies.parquet"
        if not unified_path.exists():
            logger.warning("统一电影表不存在，使用movies.parquet")
            unified_path = self.processed_dir / "movies.parquet"
        movies = pd.read_parquet(unified_path)

        # 构建文本：标题 + 类型 + 概述
        texts = []
        for _, row in movies.iterrows():
            parts = [str(row.get("title", ""))]
            if "genres" in row and pd.notna(row["genres"]):
                parts.append(str(row["genres"]))
            overview = row.get("overview", "")
            if pd.notna(overview) and str(overview).strip():
                parts.append(str(overview))
            texts.append(" ".join(parts))

        logger.info(f"编码 {len(texts)} 部电影的文本特征...")
        self.movie_embeddings = self.bert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2归一化以便余弦相似度
        )

        # 保存
        np.save(cache_path, self.movie_embeddings)
        # 同时保存对应的movie_idx列表
        if "movie_idx" in movies.columns:
            np.save(
                self.embeddings_dir / "movie_bert_ids.npy",
                movies["movie_idx"].values,
            )

        logger.info(f"BERT嵌入完成: shape={self.movie_embeddings.shape}")
        return self.movie_embeddings

    def build_knowledge_graph_triples(self) -> list:
        """从IMDB数据构建知识图谱三元组 (head, relation, tail)"""
        unified_path = self.processed_dir / "unified_movies.parquet"
        if not unified_path.exists():
            logger.warning("统一电影表不存在，跳过KG构建")
            return []

        movies = pd.read_parquet(unified_path)
        triples = []

        for _, row in tqdm(movies.iterrows(), total=len(movies), desc="构建KG三元组"):
            movie_title = str(row.get("title", ""))
            if not movie_title:
                continue

            # 电影 —[has_genre]→ 类型
            genres = str(row.get("genres", ""))
            for g in genres.split("|"):
                g = g.strip()
                if g:
                    triples.append((movie_title, "has_genre", g))

            # 电影 —[directed_by]→ 导演 / —[acted_by]→ 演员
            crew = str(row.get("crew_names", ""))
            for name in crew.split("|"):
                name = name.strip()
                if name:
                    triples.append((movie_title, "has_crew", name))

            actors = str(row.get("actors", ""))
            for name in actors.split("|"):
                name = name.strip()
                if name:
                    triples.append((movie_title, "has_actor", name))

            # 电影 —[released_in]→ 年份
            year = row.get("startYear")
            if pd.notna(year):
                triples.append((movie_title, "released_in", str(int(year))))

        logger.info(f"KG三元组构建完成: {len(triples)}条")
        # 保存
        import json
        kg_path = self.processed_dir / "kg_triples.json"
        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)

        return triples

    def build_genre_features(self) -> np.ndarray:
        """构建电影类型多热编码矩阵"""
        movies = pd.read_parquet(self.processed_dir / "movies.parquet")

        # 收集所有类型
        all_genres = set()
        genre_lists = []
        for g_str in movies["genres"]:
            g_list = str(g_str).split("|") if pd.notna(g_str) else []
            genre_lists.append(g_list)
            all_genres.update(g_list)

        all_genres = sorted(all_genres)
        genre_to_idx = {g: i for i, g in enumerate(all_genres)}

        # 构建多热矩阵
        n_movies = len(movies)
        genre_matrix = np.zeros((n_movies, len(all_genres)), dtype=np.float32)
        for i, g_list in enumerate(genre_lists):
            for g in g_list:
                if g in genre_to_idx:
                    genre_matrix[i, genre_to_idx[g]] = 1.0

        np.save(self.embeddings_dir / "genre_matrix.npy", genre_matrix)
        np.save(self.embeddings_dir / "genre_names.npy", np.array(all_genres))
        logger.info(f"类型特征矩阵: {genre_matrix.shape}")
        return genre_matrix

    def find_similar_movies(self, movie_idx: int, top_k: int = 10) -> list:
        """基于BERT嵌入查找最相似电影"""
        if self.movie_embeddings is None:
            self.build_bert_embeddings()

        query_vec = self.movie_embeddings[movie_idx]
        similarities = self.movie_embeddings @ query_vec  # 已归一化

        # 排除自身
        top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]
        return [(int(i), float(similarities[i])) for i in top_indices]
