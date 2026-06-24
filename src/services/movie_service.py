"""电影数据服务 — 统一的电影查找、搜索、相似推荐"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from .tmdb_service import TMDBService

logger = logging.getLogger(__name__)


class MovieService:
    """
    电影数据服务 — 将unified_movies.parquet与TMDB富化结合。

    所有电影查找以 movie_idx 为主键。
    """

    def __init__(self, tmdb_service: TMDBService = None):
        self.tmdb = tmdb_service
        self.movies_df: Optional[pd.DataFrame] = None
        self._feature_engineer = None  # 延迟加载

    # ── data loading ───────────────────────────────────────────────

    def load(self) -> None:
        """加载统一电影数据"""
        processed_dir = Config.get_project_root() / "data/processed"
        unified_path = processed_dir / "unified_movies.parquet"
        movies_path = processed_dir / "movies.parquet"

        if unified_path.exists():
            self.movies_df = pd.read_parquet(unified_path)
            logger.info(f"已加载 unified_movies: {len(self.movies_df)} 部电影")
        elif movies_path.exists():
            self.movies_df = pd.read_parquet(movies_path)
            logger.info(f"已加载 movies (fallback): {len(self.movies_df)} 部电影")
        else:
            self.movies_df = pd.DataFrame()
            logger.warning("未找到电影数据文件")

    @property
    def feature_engineer(self):
        """延迟加载FeatureEngineer (BERT嵌入28MB)"""
        if self._feature_engineer is None:
            from ..data.feature_engineer import FeatureEngineer
            self._feature_engineer = FeatureEngineer()
            self._feature_engineer.build_bert_embeddings()
        return self._feature_engineer

    # ── column resolution helpers ──────────────────────────────────

    @property
    def title_col(self) -> str:
        return "title" if "title" in self.movies_df.columns else "primaryTitle"

    @property
    def genre_col(self) -> Optional[str]:
        for col in ["genres", "genres_ml"]:
            if col in self.movies_df.columns:
                return col
        return None

    @property
    def rating_col(self) -> Optional[str]:
        for col in ["averageRating", "vote_average", "rating"]:
            if col in self.movies_df.columns:
                return col
        return None

    # ── row → dict conversion ──────────────────────────────────────

    def _row_to_card_dict(self, row: pd.Series) -> dict:
        """DataFrame行 → MovieCard字典"""
        import re
        imdb_id = str(row.get("imdb_id", "")) if pd.notna(row.get("imdb_id")) else None

        # 基本字段 — 清理标题（MovieLens标题自带年份如 "Star Wars (1977)"）
        raw_title = str(row.get(self.title_col, ""))
        title = re.sub(r'\s*\(\d{4}\)\s*$', '', raw_title).strip()  # 去掉末尾 (YYYY)

        card = {
            "movie_idx": int(row["movie_idx"]),
            "movie_id": int(row.get("movieId", 0)),
            "title": title,
            "year": (
                int(row["startYear"])
                if pd.notna(row.get("startYear")) and row.get("startYear") != "\\N"
                else None
            ),
            "genres": str(row.get(self.genre_col, "")),
            "rating": (
                float(row[self.rating_col])
                if self.rating_col and pd.notna(row.get(self.rating_col))
                else None
            ),
            "imdb_id": imdb_id,
            "tmdb_id": None,
            "poster_url": None,
            "overview_short": None,
        }

        # TMDB富化
        if self.tmdb and imdb_id:
            enrichment = self.tmdb.enrich_card(imdb_id)
            card["tmdb_id"] = enrichment.get("tmdb_id")
            card["poster_url"] = enrichment.get("poster_url")
            card["overview_short"] = enrichment.get("overview_short")

        return card

    def _row_to_detail_dict(self, row: pd.Series) -> dict:
        """DataFrame行 → MovieDetail字典 (含TMDB完整数据)"""
        imdb_id = str(row.get("imdb_id", "")) if pd.notna(row.get("imdb_id")) else None
        card = self._row_to_card_dict(row)

        # IMDB元数据
        card.update({
            "runtime": (
                int(row["runtimeMinutes"])
                if pd.notna(row.get("runtimeMinutes")) and row.get("runtimeMinutes") != "\\N"
                else None
            ),
            "vote_count": (
                int(row["numVotes"])
                if pd.notna(row.get("numVotes"))
                else None
            ),
            "actors": str(row.get("actors", "")) if pd.notna(row.get("actors")) else None,
            "director": str(row.get("crew_names", "")) if pd.notna(row.get("crew_names")) else None,
            "tagline": None,
            "backdrop_url": None,
            "overview": None,
            "cast": [],
            "trailers": [],
            "watch_providers": {},
            "tmdb_url": None,
            "imdb_url": f"https://www.imdb.com/title/{imdb_id}" if imdb_id else None,
        })

        # TMDB完整富化 (含视频、提供商、演职人员)
        if self.tmdb and imdb_id:
            full = self.tmdb.get_full_enrichment(imdb_id)
            if full:
                card["tmdb_id"] = full.get("tmdb_id")
                card["overview"] = full.get("overview", "")
                card["poster_url"] = TMDBService.get_poster_url(full.get("poster_path", ""), "w500")
                card["backdrop_url"] = TMDBService.get_backdrop_url(full.get("backdrop_path", ""))
                card["tagline"] = full.get("tagline", "")
                card["runtime"] = full.get("runtime") or card["runtime"]
                card["rating"] = full.get("vote_average") or card["rating"]
                card["tmdb_url"] = TMDBService.get_tmdb_web_url(full["tmdb_id"]) if full.get("tmdb_id") else None

                # 预告片
                videos = full.get("videos", [])
                card["trailers"] = [
                    {
                        "key": v["key"],
                        "name": v.get("name", ""),
                        "site": v.get("site", ""),
                        "type": v.get("type", ""),
                        "embed_url": TMDBService.get_youtube_embed_url(v["key"]),
                        "watch_url": TMDBService.get_youtube_url(v["key"]),
                    }
                    for v in videos
                    if v.get("type") in ("Trailer", "Teaser")
                ]

                # 演职人员
                credits = full.get("credits", {})
                card["cast"] = [
                    {
                        "name": c["name"],
                        "character": c.get("character", ""),
                        "profile_url": TMDBService.get_profile_url(c.get("profile_path", "")),
                    }
                    for c in credits.get("cast", [])
                ]
                directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
                card["director"] = ", ".join(directors) if directors else card["director"]

                # 观看提供商
                card["watch_providers"] = full.get("watch_providers", {})

        # ── 构建外部观影链接 ────────────────────────────────────
        title = card.get("title", "")
        year = card.get("year")
        card["external_watch_links"] = []
        card["provider_watch_url"] = None

        if title:
            link_defs = [
                ("justwatch", "JustWatch 全网搜索", "aggregator", "🔍",
                 TMDBService.get_justwatch_search_url(title, year)),
                ("tubi", "Tubi (免费)", "free", "",
                 TMDBService.get_tubi_search_url(title)),
                ("plutotv", "Pluto TV (免费)", "free", "",
                 TMDBService.get_plutotv_search_url(title)),
                ("youtube", "YouTube 完整电影", "free", "",
                 TMDBService.get_youtube_full_movie_url(title, year)),
                ("bilibili", "Bilibili", "chinese", "",
                 TMDBService.get_bilibili_search_url(title)),
                ("iqiyi", "爱奇艺", "chinese", "",
                 TMDBService.get_iqiyi_search_url(title)),
                ("tencent", "腾讯视频", "chinese", "",
                 TMDBService.get_tencent_video_search_url(title)),
                ("archive", "Internet Archive 公共领域", "public_domain", "📚",
                 TMDBService.get_internet_archive_url(title, year)),
            ]
            for platform, label, category, icon, url in link_defs:
                card["external_watch_links"].append({
                    "platform": platform,
                    "url": url,
                    "label": label,
                    "category": category,
                    "icon": icon,
                })

        if card.get("tmdb_id"):
            card["provider_watch_url"] = TMDBService.get_tmdb_watch_url(card["tmdb_id"])

        return card

    # ── query methods ──────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        """全文搜索电影 (标题 + 类型)"""
        if self.movies_df is None or len(self.movies_df) == 0:
            return []

        df = self.movies_df.copy()
        title_col = self.title_col

        # 在标题中搜索
        mask = df[title_col].str.contains(query, case=False, na=False)

        # 也在类型中搜索
        genre_col = self.genre_col
        if genre_col:
            mask |= df[genre_col].str.contains(query, case=False, na=False)

        results = df[mask]

        # 按评分排序 (如果有关联度判断)
        rating_col = self.rating_col
        if rating_col and len(results) > top_k:
            results = results.sort_values(rating_col, ascending=False)

        results = results.head(top_k)
        return [self._row_to_card_dict(row) for _, row in results.iterrows()]

    def get_detail(self, movie_idx: int) -> Optional[dict]:
        """获取电影完整详情"""
        if self.movies_df is None:
            return None
        rows = self.movies_df[self.movies_df["movie_idx"] == movie_idx]
        if len(rows) == 0:
            return None
        return self._row_to_detail_dict(rows.iloc[0])

    def get_similar(self, movie_idx: int, top_k: int = 8) -> List[dict]:
        """基于BERT嵌入的相似电影推荐"""
        if self.movies_df is None:
            return []

        try:
            fe = self.feature_engineer
            similar = fe.find_similar_movies(movie_idx, top_k=top_k)
        except Exception as e:
            logger.warning(f"BERT相似度查询失败: {e}")
            return []

        cards = []
        for mid, sim in similar:
            rows = self.movies_df[self.movies_df["movie_idx"] == mid]
            if len(rows) > 0:
                card = self._row_to_card_dict(rows.iloc[0])
                card["similarity"] = round(sim, 3)
                cards.append(card)
        return cards

    def get_popular(self, top_k: int = 20) -> List[dict]:
        """获取评分最高/最热门的电影"""
        if self.movies_df is None:
            return []

        df = self.movies_df.copy()
        rating_col = self.rating_col
        if rating_col:
            # 至少有一定数量的评分
            if "numVotes" in df.columns:
                df = df[df["numVotes"] >= 100]
            df = df.sort_values(rating_col, ascending=False)

        results = df.head(top_k)
        return [self._row_to_card_dict(row) for _, row in results.iterrows()]

    def get_by_genre(self, genre: str, top_k: int = 20) -> List[dict]:
        """按类型筛选电影"""
        if self.movies_df is None:
            return []

        df = self.movies_df.copy()
        genre_col = self.genre_col
        if genre_col:
            df = df[df[genre_col].str.contains(genre, case=False, na=False)]

        rating_col = self.rating_col
        if rating_col:
            df = df.sort_values(rating_col, ascending=False)

        results = df.head(top_k)
        return [self._row_to_card_dict(row) for _, row in results.iterrows()]

    def get_all_genres(self) -> List[str]:
        """获取所有类型列表"""
        if self.movies_df is None:
            return []

        genre_col = self.genre_col
        if not genre_col:
            return []

        all_genres = set()
        for g_str in self.movies_df[genre_col].dropna():
            sep = "|" if "|" in str(g_str) else ","
            for g in str(g_str).split(sep):
                g = g.strip()
                if g:
                    all_genres.add(g)

        return sorted(all_genres)

    def get_by_ids(self, movie_indices: List[int]) -> List[dict]:
        """批量按movie_idx获取电影卡片"""
        if self.movies_df is None:
            return []

        df = self.movies_df[self.movies_df["movie_idx"].isin(movie_indices)]
        return [self._row_to_card_dict(row) for _, row in df.iterrows()]
