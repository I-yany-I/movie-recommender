"""页面路由 — Jinja2模板渲染"""
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """首页 — 热门电影 + Hero搜索"""
    movies = request.app.state.movies
    popular = movies.get_popular(20)
    genres = movies.get_all_genres()[:12]  # 前12个类型用于导航
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "popular": popular,
            "genres": genres,
        },
    )


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = Query("")):
    """搜索页 — 搜索结果网格"""
    movies = request.app.state.movies
    results = movies.search(q, 40) if q else []
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="search.html",
        context={
            "query": q,
            "results": results,
            "total": len(results),
        },
    )


@router.get("/movie/{movie_idx}", response_class=HTMLResponse)
async def movie_detail_page(request: Request, movie_idx: int):
    """电影详情页 — 完整信息 + 预告片 + 相似推荐"""
    from fastapi.responses import RedirectResponse

    movies = request.app.state.movies
    detail = movies.get_detail(movie_idx)
    if detail is None:
        return RedirectResponse("/search?q=", status_code=302)

    similar = movies.get_similar(movie_idx, 8)
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="movie_detail.html",
        context={
            "movie": detail,
            "similar": similar,
        },
    )


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """对话页 — 聊天界面"""
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="chat.html",
    )


@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """关于页"""
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="base.html",
        context={
            "page_title": "关于 CinemaScope",
        },
    )
