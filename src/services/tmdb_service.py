"""TMDB API 服务 — 电影元数据、预告片、观看提供商、演职人员"""
import json
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from ..config import Config

logger = logging.getLogger(__name__)


class TMDBService:
    """
    TMDB API 增强服务，基于JSON文件缓存。

    支持的三种观影方式:
    1. YouTube预告片嵌入 (via /movie/{id}/videos)
    2. 流媒体平台跳转 (via /movie/{id}/watch/providers)
    3. TMDB详情页链接 (https://www.themoviedb.org/movie/{id})

    Cache结构 (tmdb_cache.json):
    {
      "tt0111161": {
        "tmdb_id": 278,
        "overview": "...",
        "poster_path": "/...jpg",
        "backdrop_path": "/...jpg",
        "vote_average": 8.7,
        "vote_count": 25000,
        "release_date": "1994-09-23",
        "runtime": 142,
        "tagline": "...",
        "videos": [{"key": "...", "name": "Trailer", "site": "YouTube", "type": "Trailer"}],
        "watch_providers": {"CN": {"flatrate": [...], "rent": [...], "buy": [...]}},
        "credits": {"cast": [...], "crew": [...]}
      }
    }
    """

    IMAGE_BASE = "https://image.tmdb.org/t/p"
    TMDB_BASE = "https://api.themoviedb.org/3"
    WEB_BASE = "https://www.themoviedb.org/movie"

    def __init__(self, api_key: str = None, cache_path: str = None, proxy: str = None):
        self.api_key = api_key or Config.get("data.tmdb.api_key", "")
        if cache_path:
            self.cache_path = Path(cache_path)
        else:
            self.cache_path = Config.get_project_root() / Config.get(
                "data.tmdb.cache_path", "data/raw/tmdb_cache.json"
            )
        self.delay = Config.get("data.tmdb.request_delay", 0.25)
        self.cache: dict = {}
        self._session: Optional[requests.Session] = None

        # 代理配置：优先用传入参数 → 环境变量 → 系统默认
        self.proxy = proxy or Config.get("data.tmdb.proxy", None)
        if not self.proxy:
            # requests库自动读取 HTTP_PROXY / HTTPS_PROXY 环境变量
            self.proxy = None  # None = 使用系统默认

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    # ── cache management ──────────────────────────────────────────

    def load_cache(self) -> dict:
        """加载TMDB缓存文件"""
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            logger.info(f"TMDB缓存已加载: {len(self.cache)} 部电影")
        else:
            self.cache = {}
            logger.info("TMDB缓存文件不存在，将从头构建")
        return self.cache

    def save_cache(self) -> None:
        """保存TMDB缓存到文件"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        logger.info(f"TMDB缓存已保存: {len(self.cache)} 部电影")

    # ── API helpers ───────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """通用TMDB API GET请求 (短超时，不阻塞页面)"""
        if not self.api_key:
            return None
        params = params or {}
        params["api_key"] = self.api_key
        url = f"{self.TMDB_BASE}{endpoint}"
        try:
            proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
            resp = self.session.get(url, params=params, timeout=(5, 8), proxies=proxies)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            else:
                logger.debug(f"TMDB API {endpoint} 返回 {resp.status_code}")
                return None
        except Exception as e:
            logger.debug(f"TMDB API请求失败 (将使用缓存数据): {endpoint}")
            return None

    def test_connection(self) -> bool:
        """测试TMDB API连通性"""
        try:
            proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
            resp = self.session.get(
                f"{self.TMDB_BASE}/configuration",
                params={"api_key": self.api_key},
                timeout=(5, 5),
                proxies=proxies,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── basic enrichment (by IMDB ID) ─────────────────────────────

    def find_by_imdb(self, imdb_id: str) -> Optional[dict]:
        """通过IMDB ID查找TMDB电影 (使用/find端点) — 结果缓存到JSON文件"""
        if not imdb_id:
            return None
        if imdb_id in self.cache:
            return self.cache[imdb_id]

        data = self._get(f"/find/{imdb_id}", {"external_source": "imdb_id"})
        if data and data.get("movie_results"):
            movie = data["movie_results"][0]
            entry = {
                "tmdb_id": movie["id"],
                "overview": movie.get("overview", ""),
                "poster_path": movie.get("poster_path", ""),
                "backdrop_path": movie.get("backdrop_path", ""),
                "vote_average": movie.get("vote_average"),
                "vote_count": movie.get("vote_count"),
                "release_date": movie.get("release_date", ""),
            }
            self.cache[imdb_id] = entry
            time.sleep(self.delay)
            return entry

        # API请求失败时，标记为空但后续仍会重试（不缓存失败结果）
        return None

    def batch_fetch_basic(self, imdb_ids: list, max_count: int = None) -> int:
        """批量获取基本TMDB元数据，返回新增数量"""
        self.load_cache()
        to_fetch = [iid for iid in imdb_ids if iid and iid not in self.cache]
        if max_count:
            to_fetch = to_fetch[:max_count]

        fetched = 0
        total = len(to_fetch)
        logger.info(f"开始TMDB批量获取: {total} 部电影待处理")

        for i, imdb_id in enumerate(to_fetch):
            if i > 0 and i % 50 == 0:
                logger.info(f"  进度: {i}/{total} ({fetched} 成功)")
                self.save_cache()  # 每50个保存一次

            self.find_by_imdb(imdb_id)
            if self.cache.get(imdb_id) is not None:
                fetched += 1

        self.save_cache()
        logger.info(f"TMDB批量获取完成: {fetched}/{total} 部电影")
        return fetched

    # ── full enrichment (videos + providers + credits) ────────────

    def get_full_enrichment(self, imdb_id: str) -> Optional[dict]:
        """获取完整TMDB数据 (含视频、提供商、演职人员)，带缓存"""
        # 先确保基本数据存在
        entry = self.cache.get(imdb_id)
        if entry is None:
            entry = self.find_by_imdb(imdb_id)
        if entry is None:
            return None

        tmdb_id = entry["tmdb_id"]

        # 如果已经获取过完整数据，直接返回
        if "videos" in entry:
            return entry

        # 使用append_to_response一次获取多个子资源
        data = self._get(
            f"/movie/{tmdb_id}",
            {"append_to_response": "videos,watch/providers,credits"},
        )
        if data:
            entry["overview"] = data.get("overview", entry.get("overview", ""))
            entry["poster_path"] = data.get("poster_path", entry.get("poster_path", ""))
            entry["backdrop_path"] = data.get("backdrop_path", entry.get("backdrop_path", ""))
            entry["vote_average"] = data.get("vote_average", entry.get("vote_average"))
            entry["vote_count"] = data.get("vote_count", entry.get("vote_count"))
            entry["release_date"] = data.get("release_date", entry.get("release_date", ""))
            entry["runtime"] = data.get("runtime")
            entry["tagline"] = data.get("tagline", "")

            # 视频 (过滤YouTube预告片)
            videos = data.get("videos", {}).get("results", [])
            entry["videos"] = [
                {
                    "key": v["key"],
                    "name": v.get("name", ""),
                    "site": v.get("site", ""),
                    "type": v.get("type", ""),
                    "official": v.get("official", False),
                }
                for v in videos
                if v.get("site") == "YouTube"
            ]

            # 观看提供商 (取中国/美国/香港)
            providers = data.get("watch/providers", {}).get("results", {})
            entry["watch_providers"] = {}
            for region in ["CN", "US", "HK"]:
                if region in providers:
                    region_data = providers[region]
                    entry["watch_providers"][region] = {
                        "link": region_data.get("link", ""),  # TMDB watch page link
                    }
                    for ptype in ("flatrate", "rent", "buy"):
                        if ptype in region_data and region_data[ptype]:
                            entry["watch_providers"][region][ptype] = [
                                {
                                    "name": p["provider_name"],
                                    "logo_path": p.get("logo_path", ""),
                                    "provider_id": p["provider_id"],
                                }
                                for p in region_data[ptype]
                            ]

            # 演职人员 (导演 + 前10演员)
            credits = data.get("credits", {})
            entry["credits"] = {
                "cast": [
                    {
                        "name": c["name"],
                        "character": c.get("character", ""),
                        "profile_path": c.get("profile_path", ""),
                    }
                    for c in credits.get("cast", [])[:10]
                ],
                "crew": [
                    {
                        "name": c["name"],
                        "job": c.get("job", ""),
                    }
                    for c in credits.get("crew", [])
                    if c.get("job") == "Director"
                ],
            }

            self.cache[imdb_id] = entry
            time.sleep(self.delay)

        return entry

    # ── image URL helpers ─────────────────────────────────────────

    @staticmethod
    def get_poster_url(poster_path: str, size: str = "w342") -> Optional[str]:
        """构建海报图片URL"""
        if not poster_path:
            return None
        return f"{TMDBService.IMAGE_BASE}/{size}{poster_path}"

    @staticmethod
    def get_backdrop_url(backdrop_path: str, size: str = "w1280") -> Optional[str]:
        """构建背景图URL"""
        if not backdrop_path:
            return None
        return f"{TMDBService.IMAGE_BASE}/{size}{backdrop_path}"

    @staticmethod
    def get_profile_url(profile_path: str, size: str = "w185") -> Optional[str]:
        """构建演员头像URL"""
        if not profile_path:
            return None
        return f"{TMDBService.IMAGE_BASE}/{size}{profile_path}"

    @staticmethod
    def get_logo_url(logo_path: str, size: str = "original") -> Optional[str]:
        """构建提供商logo URL"""
        if not logo_path:
            return None
        return f"{TMDBService.IMAGE_BASE}/{size}{logo_path}"

    @staticmethod
    def get_tmdb_web_url(tmdb_id: int) -> str:
        """构建TMDB网站电影详情页URL"""
        return f"{TMDBService.WEB_BASE}/{tmdb_id}"

    @staticmethod
    def get_youtube_url(youtube_key: str) -> str:
        """构建YouTube观看链接"""
        return f"https://www.youtube.com/watch?v={youtube_key}"

    @staticmethod
    def get_youtube_embed_url(youtube_key: str) -> str:
        """构建YouTube嵌入链接"""
        return f"https://www.youtube.com/embed/{youtube_key}"

    @staticmethod
    def get_imdb_url(imdb_id: str) -> str:
        """构建IMDB电影详情页URL"""
        return f"https://www.imdb.com/title/{imdb_id}"

    @staticmethod
    def get_tmdb_watch_url(tmdb_id: int) -> str:
        """构建TMDB '在哪里看'页面URL"""
        return f"{TMDBService.WEB_BASE}/{tmdb_id}/watch"

    # ── external watch link builders ──────────────────────────────

    @staticmethod
    def get_justwatch_search_url(title: str, year: int = None) -> str:
        """JustWatch 全网搜索"""
        query = title if not year else f"{title} {year}"
        return f"https://www.justwatch.com/cn/search?q={quote(query)}"

    @staticmethod
    def get_tubi_search_url(title: str) -> str:
        """Tubi 免费流媒体搜索"""
        return f"https://tubitv.com/search?q={quote(title)}"

    @staticmethod
    def get_plutotv_search_url(title: str) -> str:
        """Pluto TV 免费流媒体搜索"""
        return f"https://pluto.tv/search?q={quote(title)}"

    @staticmethod
    def get_youtube_full_movie_url(title: str, year: int = None) -> str:
        """YouTube 完整电影搜索"""
        query = f"{title} {year} full movie" if year else f"{title} full movie"
        return f"https://www.youtube.com/results?search_query={quote(query)}"

    @staticmethod
    def get_internet_archive_url(title: str, year: int = None) -> str:
        """Internet Archive 公共领域电影搜索"""
        query = title if not year else f"{title} {year}"
        return f"https://archive.org/search.php?query={quote(query)}+movie"

    @staticmethod
    def get_bilibili_search_url(title: str) -> str:
        """B站搜索"""
        return f"https://search.bilibili.com/all?keyword={quote(title)}"

    @staticmethod
    def get_iqiyi_search_url(title: str) -> str:
        """爱奇艺搜索"""
        return f"https://so.iqiyi.com/so/q_{quote(title)}"

    @staticmethod
    def get_tencent_video_search_url(title: str) -> str:
        """腾讯视频搜索"""
        return f"https://v.qq.com/x/search/?q={quote(title)}"

    # ── card enrichment (lightweight, for list displays) ──────────

    def enrich_card(self, imdb_id: str) -> dict:
        """为电影卡片获取轻量TMDB数据（海报+评分）"""
        entry = self.cache.get(imdb_id)
        if entry is None and imdb_id:
            entry = self.find_by_imdb(imdb_id)

        if entry:
            return {
                "poster_url": self.get_poster_url(entry.get("poster_path", "")),
                "tmdb_rating": entry.get("vote_average"),
                "tmdb_id": entry.get("tmdb_id"),
                "overview_short": (entry.get("overview", "") or "")[:120],
            }
        return {
            "poster_url": None,
            "tmdb_rating": None,
            "tmdb_id": None,
            "overview_short": None,
        }
