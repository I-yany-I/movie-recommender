"""聊天API路由 — POST /api/chat 和 GET /api/chat/stream (SSE)"""
import json
import logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """
    处理聊天消息，返回结构化卡片响应（非流式）。

    请求: {"message": "推荐科幻片", "history": [...]}
    响应: {"reply": "为你推荐...", "cards": [...], "has_cards": true}
    """
    chat_service = request.app.state.chat
    result = chat_service.chat(body.message, body.history)

    from ..schemas import MovieCard
    cards = [MovieCard(**c) for c in result.get("cards", [])]

    return ChatResponse(
        reply=result.get("reply", ""),
        cards=cards,
        has_cards=len(cards) > 0,
    )


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    message: str = "",
):
    """
    SSE流式对话 — 逐token返回文本（体验接近ChatGPT）。

    前端用法:
        const eventSource = new EventSource('/api/chat/stream?message=推荐科幻片');
        eventSource.onmessage = (e) => { appendText(e.data); };
        eventSource.onerror = () => { eventSource.close(); };
    """
    if not message:
        return StreamingResponse(
            iter(["data: 请输入消息内容\n\n"]),
            media_type="text/event-stream",
        )

    chat_service = request.app.state.chat

    async def generate():
        try:
            for token in chat_service.chat_stream(message):
                # SSE 格式: data: <content>\n\n
                escaped = json.dumps(token, ensure_ascii=False)
                yield f"data: {escaped}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"SSE流错误: {e}")
            yield f"data: {json.dumps(f'[错误] {e}', ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
