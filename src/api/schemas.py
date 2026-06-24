"""Pydantic schemas — API请求/响应的数据结构"""
from typing import List, Optional
from pydantic import BaseModel


# ── 电影卡片（列表展示用）──────────────────────────────────

class MovieCard(BaseModel):
    """电影卡片 — 用于搜索结果、推荐列表、聊天卡片"""
    movie_idx: int
    movie_id: int = 0
    title: str
    year: Optional[int] = None
    genres: str = ""
    rating: Optional[float] = None
    imdb_id: Optional[str] = None
    tmdb_id: Optional[int] = None
    poster_url: Optional[str] = None
    overview_short: Optional[str] = None
    similarity: Optional[float] = None  # BERT相似度


# ── 电影详情 ──────────────────────────────────────────────

class CastMember(BaseModel):
    name: str
    character: str = ""
    profile_url: Optional[str] = None


class TrailerInfo(BaseModel):
    key: str
    name: str = ""
    site: str = "YouTube"
    type: str = ""
    embed_url: Optional[str] = None
    watch_url: Optional[str] = None


class ProviderInfo(BaseModel):
    name: str
    logo_url: Optional[str] = None
    provider_type: str = ""  # flatrate, rent, buy


class WatchLink(BaseModel):
    """外部观影链接 — JustWatch/免费平台/中文平台/公共领域"""
    platform: str
    url: str
    label: str
    category: str = ""
    icon: str = ""


class MovieDetail(BaseModel):
    """电影详情 — 用于详情页"""
    movie_idx: int
    movie_id: int = 0
    title: str
    year: Optional[int] = None
    genres: str = ""
    rating: Optional[float] = None
    vote_count: Optional[int] = None
    runtime: Optional[int] = None
    overview: Optional[str] = None
    tagline: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    imdb_id: Optional[str] = None
    tmdb_id: Optional[int] = None
    tmdb_url: Optional[str] = None
    imdb_url: Optional[str] = None
    director: Optional[str] = None
    actors: Optional[str] = None
    cast: List[CastMember] = []
    trailers: List[TrailerInfo] = []
    watch_providers: dict = {}
    external_watch_links: List[WatchLink] = []
    provider_watch_url: Optional[str] = None
    similar_movies: List[MovieCard] = []


# ── 聊天 ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []


class ChatResponse(BaseModel):
    reply: str
    cards: List[MovieCard] = []
    has_cards: bool = False


# ── 搜索 ──────────────────────────────────────────────────

class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[MovieCard]


class RecommendRequest(BaseModel):
    genres: Optional[str] = None
    style: Optional[str] = None
    top_k: int = 20
