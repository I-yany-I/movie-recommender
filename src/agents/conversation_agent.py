"""对话Agent — 基于DeepSeek LLM的自然语言推荐交互（优化版：流式+缓存）"""
import json
import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Generator

import numpy as np
import pandas as pd

from .base_agent import BaseAgent
from ..config import Config

logger = logging.getLogger(__name__)


class ConversationAgent(BaseAgent):
    """对话智能体 — 自然语言理解、推荐调度与结果整合"""

    # 类级缓存：FeatureEngineer 只加载一次
    _feature_engineer = None
    _embeddings_loaded = False

    # 查询结果缓存（LRU风格，最多200条）
    _response_cache: OrderedDict = OrderedDict()
    _CACHE_MAX = 200

    SYSTEM_PROMPT = """你是电影推荐助手"小影"。职责：
1. 理解用户的电影偏好（类型、风格、导演等）
2. 根据描述推荐合适的电影
3. 为每部推荐提供简短理由

可用工具（数据库为英文，请用英文关键词）：
- search_movies(query): 按英文关键词（标题/导演/演员）搜索电影
- recommend_by_preference(genres, style): 按类型+风格推荐。genres用英文如"Sci-Fi|Thriller"，
  style描述感觉如"mind-bending psychological complex-plot"
- get_similar_movies(movie_title): 根据电影名找相似电影

⚠️ 重要规则：
- 涉及类型偏好时必须用 recommend_by_preference，且必填 style 参数（描述电影风格和观感）
- 找相似电影时用 get_similar_movies
- 找特定导演/演员时用 search_movies
- 每次推3-5部，含中英片名、年份、类型、一句话推荐理由、评分
- 非电影问题简单回复并引导回电影话题"""

    def __init__(self, api_key: str = None):
        super().__init__(
            name="ConversationAgent",
            description="自然语言电影推荐对话Agent（流式优化版）",
        )
        self.api_key = api_key or Config.get("llm.api_key", "")
        self.model = Config.get("llm.model", "deepseek-chat")
        self.base_url = Config.get("llm.base_url", "https://api.deepseek.com")
        self.temperature = Config.get("llm.temperature", 0.7)
        self.max_tokens = Config.get("llm.max_tokens", 1024)

        # 加载数据
        self.movies_df: Optional[pd.DataFrame] = None
        self._load_data()

    # ═══════════════════════════════════════════════════════════════
    # 数据加载
    # ═══════════════════════════════════════════════════════════════

    def _load_data(self):
        """加载电影数据（仅一次）"""
        processed_dir = Config.get_project_root() / "data/processed"
        unified_path = processed_dir / "unified_movies.parquet"
        movies_path = processed_dir / "movies.parquet"

        if unified_path.exists():
            self.movies_df = pd.read_parquet(unified_path)
        elif movies_path.exists():
            self.movies_df = pd.read_parquet(movies_path)
        else:
            logger.warning("未找到电影数据，对话Agent部分功能不可用")

    @classmethod
    def _get_feature_engineer(cls):
        """获取全局唯一的FeatureEngineer实例（延迟加载+缓存）"""
        if cls._feature_engineer is None:
            from ..data.feature_engineer import FeatureEngineer
            cls._feature_engineer = FeatureEngineer()
            cls._feature_engineer.build_bert_embeddings()
            cls._embeddings_loaded = True
            logger.info("FeatureEngineer 已初始化（全局缓存）")
        elif not cls._embeddings_loaded:
            # 确保BERT嵌入已加载
            cls._feature_engineer.build_bert_embeddings()
            cls._embeddings_loaded = True
        return cls._feature_engineer

    # ═══════════════════════════════════════════════════════════════
    # 查询缓存
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def _cache_key(cls, message: str, history: List[Dict] = None) -> str:
        """生成查询缓存键"""
        raw = message + "|" + json.dumps((history or [])[-6:], ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()

    @classmethod
    def _cache_get(cls, key: str) -> Optional[str]:
        """从缓存读取（并移到末尾保持LRU）"""
        if key in cls._response_cache:
            cls._response_cache.move_to_end(key)
            return cls._response_cache[key]
        return None

    @classmethod
    def _cache_set(cls, key: str, value: str):
        """写入缓存（超出上限时淘汰最旧的）"""
        if key in cls._response_cache:
            cls._response_cache.move_to_end(key)
        cls._response_cache[key] = value
        while len(cls._response_cache) > cls._CACHE_MAX:
            cls._response_cache.popitem(last=False)

    # ═══════════════════════════════════════════════════════════════
    # 工具定义（模块级复用，避免每次创建）
    # ═══════════════════════════════════════════════════════════════

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_movies",
                "description": "根据关键词搜索电影",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "top_k": {"type": "integer", "description": "返回数量，默认5"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_by_preference",
                "description": "根据用户偏好推荐电影。必须同时提供genres和style参数。style用英文描述电影风格和观感，如'mind-bending psychological thriller with plot twists'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "genres": {"type": "string", "description": "英文类型，用'|'分隔，如'Sci-Fi|Thriller|Mystery'"},
                        "style": {"type": "string", "description": "【必填】英文描述电影风格/观感/主题，如'psychological mind-bending complex narrative'"},
                        "top_k": {"type": "integer", "description": "返回数量，默认5"},
                    },
                    "required": ["genres", "style"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_similar_movies",
                "description": "根据电影名查找相似电影",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "movie_title": {"type": "string", "description": "电影标题"},
                        "top_k": {"type": "integer", "description": "返回数量，默认5"},
                    },
                    "required": ["movie_title"],
                },
            },
        },
    ]

    # ═══════════════════════════════════════════════════════════════
    # 核心：chat（原版，兼容旧接口）
    # ═══════════════════════════════════════════════════════════════

    def _execute(self, task_spec: Dict[str, Any]) -> Any:
        user_query = task_spec.get("query", "")
        history = task_spec.get("history", [])
        return self.chat(user_query, history)

    def chat(self, user_message: str, history: List[Dict] = None) -> str:
        """处理用户消息（非流式，兼容旧调用）"""
        # 收集流式输出的全部文本
        full = []
        for chunk in self.chat_stream(user_message, history):
            full.append(chunk)
        return "".join(full)

    # ═══════════════════════════════════════════════════════════════
    # 核心：chat_stream（流式生成器）
    # ═══════════════════════════════════════════════════════════════

    def chat_stream(
        self, user_message: str, history: List[Dict] = None
    ) -> Generator[str, None, None]:
        """流式处理用户消息，逐token yield（用户只看最终回复，中间tool调用不可见）"""
        # 清理历史中的工具调用污染（来自旧版本或脏缓存）
        history = self._sanitize_history(history)

        if not self.api_key:
            yield self._fallback_response(user_message)
            return

        # 1. 查缓存
        cache_key = self._cache_key(user_message, history)
        cached = self._cache_get(cache_key)
        if cached:
            logger.info("💨 缓存命中，直接返回")
            yield cached
            return

        # 2. 构建消息
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_message})

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)

            # 3. 第一轮：判断工具 + 允许LLM输出简短引导语（如"让我搜索一下..."）
            t0 = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=512,  # 足够容纳引导语 + tool_call
            )
            msg = response.choices[0].message
            is_truncated = response.choices[0].finish_reason == "length"
            logger.info(f"第一轮API耗时: {time.time() - t0:.2f}s, truncated={is_truncated}")

            # 4. 有工具调用 → 静默执行 → 流式生成最终回复
            if msg.tool_calls:
                # 用 dict 替代 Pydantic 对象，可靠清除 content
                msg_dict = msg.model_dump(exclude_none=False)
                msg_dict['content'] = None

                # 静默执行工具
                tool_results = self._execute_tool_calls(msg.tool_calls)
                messages.append(msg_dict)
                messages.extend(tool_results)
                # 流式生成最终回复
                yield from self._stream_llm_response(client, messages)

                # 缓存最终完整响应（由外层收集后写入）
                return

            # 5. 无工具调用的情况
            if msg.content:
                if is_truncated:
                    # 被截断了 → 用流式重新生成完整回复
                    logger.info("第一轮被截断，改用流式重新生成")
                    yield from self._stream_llm_response(client, messages)
                else:
                    # 完整短回复（闲聊等），直接返回
                    clean = self._sanitize_text(msg.content)
                    self._cache_set(cache_key, clean)
                    yield clean
            else:
                # 空回复（罕见），流式重试
                yield from self._stream_llm_response(client, messages)

        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            yield self._fallback_response(user_message)
            yield fallback

    # ═══════════════════════════════════════════════════════════════
    # 流式LLM调用
    # ═══════════════════════════════════════════════════════════════

    def _stream_llm_response(
        self, client, messages: List[Dict]
    ) -> Generator[str, None, None]:
        """流式调用LLM，逐token yield"""
        full_text = []
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": False},
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_text.append(delta.content)
                    yield delta.content
        except Exception as e:
            logger.error(f"流式调用失败: {e}")
            # 降级为非流式
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                full_text.append(text)
                yield text
            except Exception:
                yield "抱歉，服务暂时不可用，请稍后再试。"

    # ═══════════════════════════════════════════════════════════════
    # 工具执行
    # ═══════════════════════════════════════════════════════════════

    def _execute_tool_calls(self, tool_calls) -> List[Dict]:
        """批量执行工具调用（单工具失败不中断其他工具，不抛异常到用户）"""
        results = []
        for tc in tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
                t0 = time.time()
                result = self._execute_tool(func_name, func_args)
                logger.info(f"工具 {func_name} 耗时: {time.time() - t0:.3f}s")
            except Exception as e:
                logger.error(f"工具 {func_name} 执行失败: {e}")
                result = {"error": f"查询失败，请稍后重试"}
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        return results

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """过滤文本中的工具调用XML标签（DeepSeek偶尔会生成这种格式）"""
        import re
        if not text:
            return text
        # 去除 <function_calls>...</function_calls> 块（含内容）
        text = re.sub(r'<function_calls>.*?</function_calls>', '', text, flags=re.DOTALL)
        # 去除 <invoke name="...">...</invoke> 块
        text = re.sub(r'<invoke[^>]*>.*?</invoke>', '', text, flags=re.DOTALL)
        # 去除孤立的 XML 标签
        text = re.sub(r'</?function_calls>', '', text)
        text = re.sub(r'</?invoke[^>]*>', '', text)
        text = re.sub(r'<parameter[^>]*>.*?</parameter>', '', text, flags=re.DOTALL)
        # 去除多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _sanitize_history(history: List[Dict]) -> List[Dict]:
        """清理对话历史中的工具调用污染（来自旧版本的脏数据）"""
        if not history:
            return history
        cleaned = []
        for msg in history:
            content = msg.get('content', '')
            if isinstance(content, str) and content:
                sanitized = ConversationAgent._sanitize_text(content)
                if sanitized:  # 过滤后为空就丢弃整条消息
                    cleaned.append({**msg, 'content': sanitized})
            elif content is None and msg.get('role') == 'assistant':
                # tool_calls 消息（content=None），保留
                cleaned.append(msg)
            else:
                cleaned.append(msg)
        return cleaned

    def _execute_tool(self, func_name: str, args: Dict) -> Dict:
        """路由到具体工具函数"""
        if func_name == "search_movies":
            return self._tool_search_movies(args.get("query", ""), args.get("top_k", 5))
        elif func_name == "recommend_by_preference":
            return self._tool_recommend_by_preference(
                args.get("genres", ""), args.get("style", ""), args.get("top_k", 5)
            )
        elif func_name == "get_similar_movies":
            return self._tool_get_similar_movies(
                args.get("movie_title", ""), args.get("top_k", 5)
            )
        return {"error": f"未知工具: {func_name}"}

    # ═══════════════════════════════════════════════════════════════
    # 工具：search_movies
    # ═══════════════════════════════════════════════════════════════

    def _build_movie_dict(self, row, title_col: str, extra: dict = None) -> Dict:
        """构建统一的电影字典"""
        rating = 0.0
        for col in ["averageRating", "vote_average", "rating"]:
            if col in row.index and pd.notna(row.get(col)):
                rating = float(row[col])
                break

        movie = {
            "movie_idx": int(row.get("movie_idx", -1)),
            "movie_id": int(row.get("movieId", 0)),
            "title": str(row.get(title_col, "")),
            "year": str(row.get("startYear", row.get("year", ""))),
            "genres": str(row.get("genres", "")),
            "rating": rating,
            "imdb_id": str(row.get("imdb_id", "")) if pd.notna(row.get("imdb_id")) else "",
        }
        if extra:
            movie.update(extra)
        return movie

    def _tool_search_movies(self, query: str, top_k: int = 5) -> Dict:
        """在电影数据库中搜索（标题、导演、演员）"""
        if self.movies_df is None:
            return {"error": "电影数据未加载", "movies": []}

        title_col = "title" if "title" in self.movies_df.columns else "primaryTitle"
        df = self.movies_df

        # 搜索标题
        mask = df[title_col].str.contains(query, case=False, na=False)
        # 搜索导演/编剧
        if "crew_names" in df.columns:
            mask |= df["crew_names"].str.contains(query, case=False, na=False)
        # 搜索演员
        if "actors" in df.columns:
            mask |= df["actors"].str.contains(query, case=False, na=False)

        # 按评分排序取top
        rating_col = None
        for col in ["averageRating", "vote_average", "rating"]:
            if col in df.columns:
                rating_col = col
                break
        if rating_col:
            results = df[mask].sort_values(rating_col, ascending=False).head(top_k)
        else:
            results = df[mask].head(top_k)

        movies = [self._build_movie_dict(row, title_col) for _, row in results.iterrows()]
        return {"query": query, "movies": movies}

    # ═══════════════════════════════════════════════════════════════
    # 工具：recommend_by_preference
    # ═══════════════════════════════════════════════════════════════

    def _tool_recommend_by_preference(self, genres: str, style: str = "", top_k: int = 5) -> Dict:
        """基于偏好推荐（优化版：评分过滤 + BERT语义 + 平衡权重）"""
        if self.movies_df is None:
            return {"error": "电影数据未加载", "movies": []}

        df = self.movies_df.copy()
        genre_list = [g.strip() for g in genres.replace("|", " ").replace("、", " ").split() if g.strip()]

        # ── 1. 确定评分列和投票数列 ──
        rating_col = None
        for col in ["averageRating", "vote_average", "rating"]:
            if col in df.columns:
                rating_col = col
                break
        votes_col = "numVotes" if "numVotes" in df.columns else None

        # ── 2. 过滤：最低评分 + 最低投票数 ──
        mask = np.ones(len(df), dtype=bool)
        if rating_col:
            mask &= df[rating_col].fillna(0) >= 6.0  # 过滤低于6分的垃圾片
        if votes_col:
            mask &= df[votes_col].fillna(0) >= 50     # 至少50人投票
        df_filtered = df[mask].reset_index(drop=True).copy()
        if len(df_filtered) == 0:
            # 放宽过滤条件
            df_filtered = df.reset_index(drop=True).copy()

        # ── 3. 计算得分 ──
        scores = np.zeros(len(df_filtered))

        # 3a. 类型匹配分（每个匹配类型 +0.3，降权避免主导）
        genre_col = "genres" if "genres" in df_filtered.columns else None
        if genre_col:
            for g in genre_list:
                matched = df_filtered[genre_col].str.contains(g, case=False, na=False).astype(float).values
                scores += matched * 0.3

        # 3b. 评分的贝叶斯加权（IMDB风格：加权平均避免冷门高分片）
        if rating_col:
            ratings = df_filtered[rating_col].fillna(0).values
            if votes_col:
                votes = df_filtered[votes_col].fillna(0).values
                C = float(df[rating_col].mean())  # 全局均值
                m = 100.0  # 最小投票阈值
                weighted_rating = (votes / (votes + m)) * ratings + (m / (votes + m)) * C
            else:
                weighted_rating = ratings
            # 归一化到 [0, 1] 区间，权重 0.7（让评分成为主导因素）
            r_min, r_max = weighted_rating.min(), weighted_rating.max()
            if r_max > r_min:
                scores += 0.7 * (weighted_rating - r_min) / (r_max - r_min)
            else:
                scores += 0.7 * 0.5

        # 3c. BERT语义匹配（如果用户提供了风格描述）
        if style and len(style.strip()) > 3:
            try:
                from sentence_transformers import SentenceTransformer
                # 复用已缓存的FeatureEngineer中的模型
                fe = self._get_feature_engineer()
                if fe.bert_model is not None:
                    # 为每部候选电影构建简短文本
                    title_col = "title" if "title" in df_filtered.columns else "primaryTitle"
                    g_col = "genres" if "genres" in df_filtered.columns else None
                    texts = []
                    for _, row in df_filtered.iterrows():
                        parts = [str(row.get(title_col, ""))]
                        if g_col and pd.notna(row.get(g_col)):
                            parts.append(str(row.get(g_col)))
                        overview = row.get("overview", "")
                        if pd.notna(overview) and str(overview).strip():
                            parts.append(str(overview)[:200])
                        texts.append(" ".join(parts))

                    # 编码并计算相似度
                    style_emb = fe.bert_model.encode(
                        [style], normalize_embeddings=True, show_progress_bar=False
                    )[0]
                    # 分批编码候选电影（避免OOM）
                    batch_size = 512
                    all_sims = []
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]
                        batch_embs = fe.bert_model.encode(
                            batch_texts, normalize_embeddings=True, show_progress_bar=False
                        )
                        batch_sims = batch_embs @ style_emb
                        all_sims.append(batch_sims)
                    bert_sims = np.concatenate(all_sims)
                    # BERT相似度权重 0.4
                    scores += 0.4 * bert_sims
                    logger.info(f"BERT语义匹配完成: {len(texts)} 部电影")
            except Exception as e:
                logger.warning(f"BERT语义匹配失败，仅用类型+评分: {e}")

        # ── 4. 排序取top ──
        top_local = np.argsort(scores)[::-1][:top_k * 3]  # 取3倍候选
        title_col = "title" if "title" in df_filtered.columns else "primaryTitle"

        # 去重（按标题去重，避免同一电影的不同版本）
        seen_titles = set()
        movies = []
        for idx in top_local:
            row = df_filtered.iloc[idx]
            title = str(row.get(title_col, ""))
            # 去掉年份后缀做去重
            import re
            clean_title = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip().lower()
            if clean_title not in seen_titles:
                seen_titles.add(clean_title)
                m = self._build_movie_dict(row, title_col)
                m["match_score"] = round(float(scores[idx]), 3)
                movies.append(m)
            if len(movies) >= top_k:
                break

        return {"genres": genres, "style": style, "movies": movies}

    # ═══════════════════════════════════════════════════════════════
    # 工具：get_similar_movies（优化版：复用FeatureEngineer）
    # ═══════════════════════════════════════════════════════════════

    def _tool_get_similar_movies(self, movie_title: str, top_k: int = 5) -> Dict:
        """获取相似电影（使用缓存的FeatureEngineer）"""
        if self.movies_df is None:
            return {"error": "电影数据未加载", "movies": []}

        title_col = "title" if "title" in self.movies_df.columns else "primaryTitle"
        # 找到目标电影
        matches = self.movies_df[
            self.movies_df[title_col].str.contains(movie_title, case=False, na=False)
        ]
        if len(matches) == 0:
            return {"error": f"未找到电影: {movie_title}", "movies": []}

        target_idx = int(matches.iloc[0].get("movie_idx", -1))
        if target_idx < 0:
            return {"error": "电影索引不可用", "movies": []}

        try:
            # 使用全局缓存的FeatureEngineer（关键优化点）
            fe = self._get_feature_engineer()
            similar = fe.find_similar_movies(target_idx, top_k=top_k)
            movies = []
            for mid, sim in similar:
                row = self.movies_df[self.movies_df["movie_idx"] == mid]
                if len(row) > 0:
                    r = row.iloc[0]
                    m = self._build_movie_dict(r, title_col)
                    m["similarity"] = round(sim, 3)
                    movies.append(m)
            return {"movie_title": movie_title, "similar_movies": movies}
        except Exception as e:
            logger.error(f"相似电影查询失败: {e}")
            return {"error": str(e), "movies": []}

    # ═══════════════════════════════════════════════════════════════
    # chat_structured（给 ChatService 用，非流式但缓存优化）
    # ═══════════════════════════════════════════════════════════════

    def chat_structured(self, user_message: str, history: List[Dict] = None) -> Dict:
        """
        处理用户消息，返回结构化响应（含电影卡片数据）。
        重要：中间工具调用过程对用户完全不可见，只返回最终结果。
        """
        # 清理历史中的工具调用污染
        history = self._sanitize_history(history)

        if not self.api_key:
            return {
                "reply": self._fallback_response(user_message),
                "tool_results": [],
                "movies": [],
            }

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_message})

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=512,  # 足够容纳引导语+tool_call（之前256太紧）
            )

            msg = response.choices[0].message
            tool_results = []
            all_movies = []

            if msg.tool_calls:
                # ⚠️ 关键：Pydantic 对象可能冻结，不能直接赋值 msg.content = None
                # 转为 dict 再清除 content，确保 100% 可靠
                msg_dict = msg.model_dump(exclude_none=False)
                msg_dict['content'] = None  # 抹掉"让我搜索一下..."等中间文本
                messages.append(msg_dict)

                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    try:
                        func_args = json.loads(tc.function.arguments)
                        result = self._execute_tool(func_name, func_args)
                    except Exception as e:
                        logger.error(f"工具 {func_name} 失败: {e}")
                        func_args = {}
                        result = {"error": "查询失败"}
                    tool_results.append({
                        "tool": func_name,
                        "args": func_args,
                        "result": result,
                    })
                    movies_key = "similar_movies" if func_name == "get_similar_movies" else "movies"
                    if movies_key in result:
                        all_movies.extend(result[movies_key])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                # 第二轮（非流式）
                final_response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                reply = final_response.choices[0].message.content or ""
                # 最终防线：正则过滤任何残留的工具调用 XML
                reply = self._sanitize_text(reply)
            else:
                reply = msg.content or "抱歉，我没有理解你的需求，能换个说法再试试吗？"
                reply = self._sanitize_text(reply)  # 无工具调用时同样过滤

            return {"reply": reply, "tool_results": tool_results, "movies": all_movies}

        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            return {
                "reply": self._fallback_response(user_message),
                "tool_results": [],
                "movies": [],
            }

    # ═══════════════════════════════════════════════════════════════
    # 离线降级
    # ═══════════════════════════════════════════════════════════════

    def _fallback_response(self, user_message: str) -> str:
        """无API时的降级回复"""
        if self.movies_df is not None:
            title_col = "title" if "title" in self.movies_df.columns else "primaryTitle"
            genre_col = "genres" if "genres" in self.movies_df.columns else None
            rating_col = None
            for col in ["averageRating", "vote_average"]:
                if col in self.movies_df.columns:
                    rating_col = col
                    break

            sample = self.movies_df
            if rating_col:
                sample = sample.sort_values(rating_col, ascending=False)

            results = sample.head(5)
            lines = ["根据当前热门评分，为你推荐以下电影：\n"]
            for i, (_, row) in enumerate(results.iterrows()):
                title = str(row.get(title_col, "未知"))
                genres = str(row.get(genre_col or "genres", ""))
                rating = float(row.get(rating_col, 0)) if rating_col else 0
                lines.append(f"{i+1}. **{title}** ({genres}) — ⭐{rating:.1f}")

            return "\n".join(lines)

        return (
            "抱歉，我暂时无法连接到AI服务。\n"
            "请确认 DEEPSEEK_API_KEY 环境变量已正确设置。\n\n"
            "或者通过命令行使用预训练模型：\n"
            "`python main.py train --model all`"
        )

    def report(self) -> str:
        cache_size = len(self._response_cache)
        fe_status = "✅ 已缓存" if self._feature_engineer is not None else "⏳ 待加载"
        return (
            "=" * 50 + "\n"
            "  💬 对话Agent — 状态报告 (优化版)\n" +
            "=" * 50 + "\n\n"
            f"  LLM模型: {self.model}\n"
            f"  API状态: {'✅ 已配置' if self.api_key else '❌ 未配置'}\n"
            f"  电影数据: {'✅ 已加载' if self.movies_df is not None else '❌ 未加载'}\n"
            f"  BERT特征: {fe_status}\n"
            f"  查询缓存: {cache_size} 条\n"
            f"  流式输出: ✅ 已启用\n\n" +
            "=" * 50
        )
