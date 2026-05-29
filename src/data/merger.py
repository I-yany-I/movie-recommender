"""多源数据融合模块 — 对齐MovieLens、IMDB、TMDB三个数据源"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from ..config import Config

logger = logging.getLogger(__name__)


class DataMerger:
    """三源数据融合器"""

    def __init__(self):
        self.project_root = Config.get_project_root()
        self.ml_path = self.project_root / Config.get("data.movielens.local_path")
        self.imdb_path = self.project_root / Config.get("data.imdb.local_path")
        self.tmdb_config = Config.get("data.tmdb", {})
        self.processed_dir = self.project_root / "data/processed"

        # 核心映射表
        self.ml_to_imdb: dict = {}       # movieId → tconst
        self.imdb_to_tmdb: dict = {}     # tconst → tmdb_id
        self.unified_movies: pd.DataFrame = None

    def run(self) -> pd.DataFrame:
        """执行融合流水线"""
        logger.info("=" * 50)
        logger.info("开始多源数据融合...")
        logger.info("=" * 50)

        self._build_ml_imdb_mapping()
        self._load_imdb_data()
        self._fetch_tmdb_data()
        self._merge_all()

        logger.info(f"融合完成: {len(self.unified_movies)}部电影已对齐")
        return self.unified_movies

    def _build_ml_imdb_mapping(self):
        """构建 MovieLens movieId → IMDB tconst 映射"""
        ml_movies = pd.read_csv(self.ml_path / "movies.csv")

        # MovieLens movies.csv 包含 imdbId 列（仅25M版本）
        links_path = self.ml_path / "links.csv"
        if links_path.exists():
            links = pd.read_csv(links_path)
            for _, row in links.iterrows():
                if pd.notna(row["imdbId"]):
                    tconst = f"tt{int(row['imdbId']):07d}"
                    self.ml_to_imdb[row["movieId"]] = tconst
            logger.info(f"构建了 {len(self.ml_to_imdb)} 条 ML→IMDB 映射")
        else:
            logger.warning("未找到 links.csv，尝试从 movies.csv imdbId 列解析")
            # 备用：从25M版本的movies.csv中可能有imdbId
            self._build_fallback_mapping()

    def _build_fallback_mapping(self):
        """备用映射方案 — 通过电影标题模糊匹配"""
        logger.info("使用电影标题+年份模糊匹配构建映射...")
        # 仅当links.csv不存在时使用
        pass

    def _load_imdb_data(self):
        """Load and parse IMDB data using chunked reads for large files."""
        logger.info("Loading IMDB datasets...")

        mapped_tconsts = set(self.ml_to_imdb.values())

        # title.basics.tsv — filter to movies we care about during read
        basics = self._read_tsv_filtered(
            self.imdb_path / "title.basics.tsv",
            key_col="tconst",
            valid_keys=mapped_tconsts,
            chunk_size=50000,
        )
        basics = basics[basics["titleType"] == "movie"].copy()
        logger.info(f"IMDB basics filtered: {len(basics)} movies")

        # title.ratings.tsv — small file, fine to load directly
        ratings = pd.read_csv(
            self.imdb_path / "title.ratings.tsv",
            sep="\t",
            na_values="\\N",
        )
        basics = basics.merge(ratings, on="tconst", how="left")
        del ratings

        # title.principals.tsv — huge, chunked read
        principals = self._read_tsv_filtered(
            self.imdb_path / "title.principals.tsv",
            key_col="tconst",
            valid_keys=mapped_tconsts,
            chunk_size=100000,
        )
        logger.info(f"IMDB principals filtered: {len(principals)} rows")

        # Collect only the nconst values present in filtered principals
        needed_nconsts = set(principals["nconst"].unique())

        # name.basics.tsv — huge, chunked read only for needed names
        names = self._read_tsv_filtered(
            self.imdb_path / "name.basics.tsv",
            key_col="nconst",
            valid_keys=needed_nconsts,
            chunk_size=100000,
        )
        logger.info(f"IMDB names filtered: {len(names)} rows")

        # Extract directors/writers
        crew = principals[principals["category"].isin(["director", "writer"])]
        crew = crew.merge(names[["nconst", "primaryName"]], on="nconst", how="left")
        crew_agg = crew.groupby("tconst").agg({
            "primaryName": lambda x: "|".join(x.dropna().unique())
        }).rename(columns={"primaryName": "crew_names"})

        # Extract top actors (first 5 per movie)
        actors = principals[principals["category"].isin(["actor", "actress"])]
        actors = actors.merge(names[["nconst", "primaryName"]], on="nconst", how="left")
        actors_agg = actors.groupby("tconst").agg({
            "primaryName": lambda x: "|".join(list(x.dropna().unique())[:5])
        }).rename(columns={"primaryName": "actors"})

        basics = basics.merge(crew_agg, on="tconst", how="left")
        basics = basics.merge(actors_agg, on="tconst", how="left")

        self.imdb_movies = basics
        logger.info(f"Loaded IMDB movies: {len(basics)}")

    @staticmethod
    def _read_tsv_filtered(filepath, key_col, valid_keys, chunk_size):
        """Read a large TSV in chunks, keeping only rows with keys in valid_keys."""
        chunks = []
        for chunk in pd.read_csv(
            filepath, sep="\t", low_memory=False,
            na_values="\\N", chunksize=chunk_size,
        ):
            filtered = chunk[chunk[key_col].isin(valid_keys)]
            if len(filtered) > 0:
                chunks.append(filtered)
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()

    def _fetch_tmdb_data(self):
        """通过TMDB API批量获取电影概述"""
        api_key = self.tmdb_config.get("api_key", "")
        if not api_key:
            logger.warning("未配置TMDB API Key，跳过TMDB数据获取")
            self.tmdb_data = {}
            return

        cache_path = self.project_root / self.tmdb_config.get("cache_path", "data/raw/tmdb_cache.json")
        cache = {}
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)

        delay = self.tmdb_config.get("request_delay", 0.25)
        base_url = self.tmdb_config.get("api_base", "https://api.themoviedb.org/3")

        imdb_ids = list(set(self.ml_to_imdb.values()))  # 去重
        logger.info(f"通过TMDB API获取 {len(imdb_ids)} 部电影的概述...")

        for tconst in tqdm(imdb_ids, desc="TMDB Fetch"):
            if tconst in cache:
                continue
            try:
                # 用IMDB ID查找TMDB条目
                url = f"{base_url}/find/{tconst}"
                resp = requests.get(
                    url,
                    params={"api_key": api_key, "external_source": "imdb_id"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    movies = data.get("movie_results", [])
                    if movies:
                        tmdb_id = movies[0]["id"]
                        cache[tconst] = {
                            "tmdb_id": tmdb_id,
                            "overview": movies[0].get("overview", ""),
                            "poster_path": movies[0].get("poster_path", ""),
                            "vote_average": movies[0].get("vote_average"),
                        }
                        self.imdb_to_tmdb[tconst] = tmdb_id
                time.sleep(delay)
            except Exception as e:
                logger.warning(f"获取 {tconst} 失败: {e}")
                continue

        # 保存缓存
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        self.tmdb_data = cache
        logger.info(f"TMDB数据获取完成: {len(cache)}部电影")

    def _merge_all(self):
        """三源融合：ML + IMDB + TMDB → 统一电影表"""
        # 1) 从processed movies.parquet读取ML电影
        ml_movies = pd.read_parquet(self.processed_dir / "movies.parquet")
        ml_movies["imdb_id"] = ml_movies["movieId"].map(self.ml_to_imdb)

        # 2) 合并IMDB信息
        if hasattr(self, "imdb_movies") and len(self.imdb_movies) > 0:
            ml_movies = ml_movies.merge(
                self.imdb_movies.rename(columns={"tconst": "imdb_id"}),
                on="imdb_id",
                how="left",
                suffixes=("_ml", ""),
            )

        # 3) 合并TMDB信息
        tmdb_rows = []
        for imdb_id, info in self.tmdb_data.items():
            tmdb_rows.append(
                {**info, "imdb_id": imdb_id}
            )
        if tmdb_rows:
            tmdb_df = pd.DataFrame(tmdb_rows)
            ml_movies = ml_movies.merge(tmdb_df, on="imdb_id", how="left")

        self.unified_movies = ml_movies

        # Fix mixed-type columns from IMDB (\\N values get read as strings)
        for col in ["runtimeMinutes", "startYear", "endYear", "isAdult"]:
            if col in self.unified_movies.columns:
                self.unified_movies[col] = pd.to_numeric(
                    self.unified_movies[col], errors="coerce"
                )

        self.unified_movies.to_parquet(self.processed_dir / "unified_movies.parquet", index=False)

        # 保存映射表
        pd.DataFrame(
            [(mid, iid) for mid, iid in self.ml_to_imdb.items()],
            columns=["movieId", "tconst"],
        ).to_parquet(self.processed_dir / "ml_imdb_mapping.parquet", index=False)

        logger.info(f"统一电影表已保存: {len(self.unified_movies)}部电影, "
                     f"{len(self.unified_movies.columns)}个特征列")
