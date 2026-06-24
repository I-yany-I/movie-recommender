"""推荐API路由"""
from fastapi import APIRouter, Request, Query

router = APIRouter(prefix="/api", tags=["recommend"])


@router.get("/recommend")
async def recommend_movies(
    request: Request,
    genres: str = Query("", description="偏好的电影类型"),
    style: str = Query("", description="风格描述"),
    top_k: int = Query(20, ge=1, le=100),
):
    """按类型和风格推荐电影"""
    rec = request.app.state.recommend
    if genres:
        results = rec.recommend_by_genres(genres, top_k)
    else:
        results = rec.recommend_popular(top_k)
    return {"genres": genres, "style": style, "total": len(results), "results": results}
