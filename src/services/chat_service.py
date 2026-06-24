"""聊天服务 — 包装ConversationAgent，支持流式输出 + 结构化卡片"""
import logging
from typing import Dict, List, Optional, Generator

from .tmdb_service import TMDBService
from .movie_service import MovieService

logger = logging.getLogger(__name__)


class ChatService:
    """
    聊天服务 — ConversationAgent + TMDB富化 → 结构化卡片/流式文本。

    优化：
    - chat_stream() 支持SSE流式输出
    - movie_indices 提取去重优化
    """

    def __init__(
        self,
        movie_service: MovieService = None,
        tmdb_service: TMDBService = None,
    ):
        self.movies = movie_service
        self.tmdb = tmdb_service
        self._agent = None

    @property
    def agent(self):
        """延迟加载ConversationAgent（全局单例）"""
        if self._agent is None:
            from ..agents.conversation_agent import ConversationAgent
            self._agent = ConversationAgent()
        return self._agent

    # ═══════════════════════════════════════════════════════════════
    # 流式对话
    # ═══════════════════════════════════════════════════════════════

    def chat_stream(
        self, message: str, history: List[Dict] = None
    ) -> Generator[str, None, None]:
        """
        流式对话 — 逐token yield文本回复。

        用法（FastAPI SSE）:
            @app.get("/api/chat/stream")
            async def chat_stream(...):
                return StreamingResponse(
                    chat_service.chat_stream(message, history),
                    media_type="text/event-stream"
                )
        """
        yield from self.agent.chat_stream(message, history)

    # ═══════════════════════════════════════════════════════════════
    # 结构化对话（非流式，含TMDB卡片）
    # ═══════════════════════════════════════════════════════════════

    def chat(self, message: str, history: List[Dict] = None) -> Dict:
        """
        处理聊天消息，返回结构化响应（含TMDB富化电影卡片）。
        """
        # 1. 通过Agent获取结构化对话结果（内部有缓存）
        structured = self.agent.chat_structured(message, history)

        # 2. 从工具结果提取movie_idx列表（去重保序）
        movie_indices = self._extract_movie_indices(structured.get("tool_results", []))

        # 3. 批量获取电影卡片（TMDB富化）
        cards = []
        if movie_indices and self.movies:
            cards = self.movies.get_by_ids(movie_indices)

        # 4. 如果没卡片但structured中有movies字段，用那个
        if not cards and structured.get("movies"):
            raw_movies = structured["movies"]
            seen = set()
            unique = []
            for m in raw_movies:
                mi = m.get("movie_idx", -1)
                if mi > 0 and mi not in seen:
                    seen.add(mi)
                    unique.append(m)
            if unique and self.movies:
                cards = self.movies.get_by_ids([m["movie_idx"] for m in unique])

        return {
            "reply": structured.get("reply", ""),
            "cards": cards,
            "has_cards": len(cards) > 0,
        }

    def _extract_movie_indices(self, tool_results: List[Dict]) -> List[int]:
        """从工具调用结果中提取movie_idx列表（去重保序）"""
        indices = []
        seen = set()
        for tr in tool_results:
            result = tr.get("result", {})
            for key in ("movies", "similar_movies"):
                movies = result.get(key, [])
                for m in movies:
                    mi = m.get("movie_idx")
                    if mi is not None and mi >= 0 and mi not in seen:
                        seen.add(mi)
                        indices.append(mi)
        return indices
