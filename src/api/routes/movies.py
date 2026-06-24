"""电影相关API路由"""
from typing import Optional
from fastapi import APIRouter, Request, Query, HTTPException

from ..schemas import MovieCard, MovieDetail, SearchResponse

router = APIRouter(prefix="/api", tags=["movies"])


@router.get("/search", response_model=SearchResponse)
async def search_movies(
    request: Request,
    q: str = Query(..., min_length=1, description="搜索关键词"),
    top_k: int = Query(20, ge=1, le=100),
):
    """搜索电影"""
    movies = request.app.state.movies
    results = movies.search(q, top_k)
    return SearchResponse(query=q, total=len(results), results=results)


@router.get("/movie/{movie_idx}", response_model=MovieDetail)
async def get_movie_detail(request: Request, movie_idx: int):
    """获取电影完整详情"""
    movies = request.app.state.movies
    detail = movies.get_detail(movie_idx)
    if detail is None:
        raise HTTPException(status_code=404, detail="电影未找到")

    # 附加相似电影
    detail["similar_movies"] = movies.get_similar(movie_idx, top_k=8)
    return detail


@router.get("/movie/{movie_idx}/similar")
async def get_similar_movies(
    request: Request,
    movie_idx: int,
    top_k: int = Query(8, ge=1, le=20),
):
    """获取相似电影列表"""
    movies = request.app.state.movies
    results = movies.get_similar(movie_idx, top_k)
    return {"movie_idx": movie_idx, "similar": results}


@router.get("/popular")
async def get_popular_movies(
    request: Request,
    top_k: int = Query(20, ge=1, le=100),
):
    """获取热门/高分电影"""
    movies = request.app.state.movies
    results = movies.get_popular(top_k)
    return {"total": len(results), "results": results}


@router.get("/genres")
async def get_all_genres(request: Request):
    """获取所有电影类型"""
    movies = request.app.state.movies
    genres = movies.get_all_genres()
    return {"genres": genres}


@router.get("/genre/{genre}")
async def get_movies_by_genre(
    request: Request,
    genre: str,
    top_k: int = Query(20, ge=1, le=100),
):
    """按类型获取电影"""
    movies = request.app.state.movies
    results = movies.get_by_genre(genre, top_k)
    return {"genre": genre, "total": len(results), "results": results}
