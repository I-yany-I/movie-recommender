"""FastAPI应用工厂 — CinemaScope Web后端"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import Config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期 — 启动时加载数据和服务"""
    logger.info("=" * 50)
    logger.info("  CinemaScope 服务器启动中...")
    logger.info("=" * 50)

    # 1. 初始化TMDB服务
    from ..services.tmdb_service import TMDBService
    tmdb_service = TMDBService()
    tmdb_service.load_cache()
    logger.info(f"TMDB缓存: {len(tmdb_service.cache)} 部电影")

    # 2. 初始化电影数据服务
    from ..services.movie_service import MovieService
    movie_service = MovieService(tmdb_service)
    movie_service.load()
    logger.info(f"电影数据: {len(movie_service.movies_df)} 部")

    # 3. 初始化推荐服务
    from ..services.recommendation_service import RecommendationService
    rec_service = RecommendationService(movie_service)
    # 延迟加载模型（太耗时，首次请求时再加载）

    # 4. 初始化聊天服务
    from ..services.chat_service import ChatService
    chat_service = ChatService(movie_service, tmdb_service)

    # 5. 初始化Jinja2模板
    templates_dir = Path(__file__).resolve().parent.parent / "web" / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    # 存储到app.state
    app.state.tmdb = tmdb_service
    app.state.movies = movie_service
    app.state.recommend = rec_service
    app.state.chat = chat_service
    app.state.templates = templates

    logger.info("  所有服务加载完成，服务器就绪")
    logger.info("=" * 50)

    yield

    # 关闭时保存缓存
    tmdb_service.save_cache()
    logger.info("CinemaScope 服务器已关闭")


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    web_config = Config.get("web", {})

    app = FastAPI(
        title=web_config.get("title", "CinemaScope"),
        description="AI-Powered Movie Discovery",
        version="2.0.0",
        lifespan=lifespan,
    )

    # 挂载静态文件
    static_dir = Path(__file__).resolve().parent.parent / "web" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # 注册API路由
    from .routes.movies import router as movies_router
    from .routes.chat import router as chat_router
    from .routes.recommend import router as recommend_router
    from .routes.pages import router as pages_router

    app.include_router(movies_router)
    app.include_router(chat_router)
    app.include_router(recommend_router)
    app.include_router(pages_router)

    return app
