"""对话Agent — 基于DeepSeek LLM的自然语言推荐交互"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_agent import BaseAgent
from ..config import Config

logger = logging.getLogger(__name__)


class ConversationAgent(BaseAgent):
    """对话智能体 — 自然语言理解、推荐调度与结果整合"""

    SYSTEM_PROMPT = """你是一个专业的电影推荐助手，名叫"小影"。你的职责是：
1. 理解用户对电影的需求和偏好（类型、风格、年代、导演等）
2. 根据用户的描述推荐合适的电影
3. 为每部推荐电影提供清晰的推荐理由
4. 如果用户对推荐不满意，耐心调整推荐策略

你有以下工具可以调用：
- search_movies(query): 根据关键词搜索电影
- recommend_by_preference(genres, style, era): 根据偏好推荐电影
- get_similar_movies(movie_title): 找相似电影

请用友好、专业的语气与用户交流。每次推荐3-5部电影，包含：
- 电影名称和年份
- 类型标签
- 简短推荐理由
- IMDB/豆瓣评分（如果有）

如果用户的问题不涉及电影推荐，可以简单回复并引导回电影话题。"""

    def __init__(self, api_key: str = None):
        super().__init__(
            name="ConversationAgent",
            description="自然语言电影推荐对话Agent",
        )
        self.api_key = api_key or Config.get("llm.api_key", "")
        self.model = Config.get("llm.model", "deepseek-chat")
        self.base_url = Config.get("llm.base_url", "https://api.deepseek.com")
        self.temperature = Config.get("llm.temperature", 0.7)
        self.max_tokens = Config.get("llm.max_tokens", 1024)

        # 加载数据用于本地查询
        self.movies_df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self):
        """加载电影数据用于工具调用"""
        processed_dir = Config.get_project_root() / "data/processed"
        unified_path = processed_dir / "unified_movies.parquet"
        movies_path = processed_dir / "movies.parquet"

        if unified_path.exists():
            self.movies_df = pd.read_parquet(unified_path)
        elif movies_path.exists():
            self.movies_df = pd.read_parquet(movies_path)
        else:
            logger.warning("未找到电影数据，对话Agent部分功能不可用")

    def _execute(self, task_spec: Dict[str, Any]) -> Any:
        """执行对话任务"""
        user_query = task_spec.get("query", "")
        history = task_spec.get("history", [])
        return self.chat(user_query, history)

    def chat(self, user_message: str, history: List[Dict] = None) -> str:
        """处理用户消息，返回推荐回复"""
        if not self.api_key:
            return self._fallback_response(user_message)

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if history:
            messages.extend(history[-10:])  # 保留最近10轮对话

        messages.append({"role": "user", "content": user_message})

        # 定义可用的 tools/functions
        tools = [
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
                    "description": "根据用户偏好推荐电影",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "genres": {"type": "string", "description": "偏好的类型，如'科幻|悬疑'"},
                            "style": {"type": "string", "description": "偏好的风格描述"},
                            "top_k": {"type": "integer", "description": "返回数量，默认5"},
                        },
                        "required": ["genres"],
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

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            msg = response.choices[0].message

            # 处理tool calls
            if msg.tool_calls:
                tool_results = []
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    func_args = json.loads(tc.function.arguments)
                    result = self._execute_tool(func_name, func_args)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                # 将工具结果发回GPT生成最终回复
                messages.append(msg)
                messages.extend(tool_results)

                final_response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return final_response.choices[0].message.content

            return msg.content or "抱歉，我没有理解你的需求，能换个说法再试试吗？"

        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            return self._fallback_response(user_message)

    def _execute_tool(self, func_name: str, args: Dict) -> Dict:
        """执行本地工具函数"""
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
        else:
            return {"error": f"未知工具: {func_name}"}

    def _tool_search_movies(self, query: str, top_k: int = 5) -> Dict:
        """在电影数据库中搜索"""
        if self.movies_df is None:
            return {"error": "电影数据未加载", "movies": []}

        title_col = "title" if "title" in self.movies_df.columns else "primaryTitle"
        df = self.movies_df.copy()
        mask = df[title_col].str.contains(query, case=False, na=False)
        results = df[mask].head(top_k)

        movies = []
        for _, row in results.iterrows():
            movies.append({
                "title": str(row.get(title_col, "")),
                "year": str(row.get("startYear", row.get("year", ""))),
                "genres": str(row.get("genres", "")),
                "rating": float(row.get("averageRating", row.get("vote_average", 0))),
            })

        return {"query": query, "movies": movies}

    def _tool_recommend_by_preference(self, genres: str, style: str = "", top_k: int = 5) -> Dict:
        """基于偏好推荐"""
        if self.movies_df is None:
            return {"error": "电影数据未加载", "movies": []}

        df = self.movies_df.copy()
        genre_list = genres.replace("|", " ").replace("、", " ").split()

        # 按类型匹配排序
        scores = np.zeros(len(df))
        genre_col = "genres" if "genres" in df.columns else None
        if genre_col:
            for g in genre_list:
                scores += df[genre_col].str.contains(g, case=False, na=False).astype(float)

        # 按评分排序
        rating_col = None
        for col in ["averageRating", "vote_average", "rating"]:
            if col in df.columns:
                rating_col = col
                break

        if rating_col:
            scores += df[rating_col].fillna(0).values / 10.0

        top_indices = np.argsort(scores)[::-1][:top_k]

        movies = []
        title_col = "title" if "title" in df.columns else "primaryTitle"
        for idx in top_indices:
            row = df.iloc[idx]
            movies.append({
                "title": str(row.get(title_col, "")),
                "year": str(row.get("startYear", "")),
                "genres": str(row.get(genre_col or "genres", "")),
                "rating": float(row.get(rating_col, 0)) if rating_col else 0.0,
            })

        return {"genres": genres, "style": style, "movies": movies}

    def _tool_get_similar_movies(self, movie_title: str, top_k: int = 5) -> Dict:
        """获取相似电影"""
        if self.movies_df is None:
            return {"error": "电影数据未加载", "movies": []}

        try:
            from ..data.feature_engineer import FeatureEngineer
            fe = FeatureEngineer()
            fe.build_bert_embeddings()

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

            similar = fe.find_similar_movies(target_idx, top_k=top_k)
            movies = []
            for mid, sim in similar:
                row = self.movies_df[self.movies_df["movie_idx"] == mid]
                if len(row) > 0:
                    r = row.iloc[0]
                    movies.append({
                        "title": str(r.get(title_col, "")),
                        "genres": str(r.get("genres", "")),
                        "similarity": round(sim, 3),
                    })

            return {"movie_title": movie_title, "similar_movies": movies}

        except Exception as e:
            logger.error(f"相似电影查询失败: {e}")
            return {"error": str(e), "movies": []}

    def _fallback_response(self, user_message: str) -> str:
        """无API时的降级回复"""
        # 用本地规则做简单推荐
        if self.movies_df is not None:
            title_col = "title" if "title" in self.movies_df.columns else "primaryTitle"
            genre_col = "genres" if "genres" in self.movies_df.columns else None
            rating_col = None
            for col in ["averageRating", "vote_average"]:
                if col in self.movies_df.columns:
                    rating_col = col
                    break

            # 尝试匹配关键词
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
        return (
            "=" * 50 + "\n"
            "  💬 对话Agent — 状态报告\n" +
            "=" * 50 + "\n\n"
            f"  LLM模型: {self.model}\n"
            f"  API状态: {'✅ 已配置' if self.api_key else '❌ 未配置'}\n"
            f"  电影数据: {'✅ 已加载' if self.movies_df is not None else '❌ 未加载'}\n\n" +
            "=" * 50
        )
