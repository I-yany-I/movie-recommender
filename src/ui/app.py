"""Gradio Web UI — 智能电影推荐助手界面"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根在path中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.agents.conversation_agent import ConversationAgent

import logging

logger = logging.getLogger(__name__)


def load_data_safe():
    """安全地加载数据，失败时返回空DataFrame"""
    processed_dir = Config.get_project_root() / "data/processed"
    unified_path = processed_dir / "unified_movies.parquet"
    movies_path = processed_dir / "movies.parquet"

    if unified_path.exists():
        return pd.read_parquet(unified_path)
    elif movies_path.exists():
        return pd.read_parquet(movies_path)
    return pd.DataFrame()


def create_chat_interface():
    """创建对话推荐Tab"""
    import gradio as gr

    conv_agent = ConversationAgent()

    def chat_fn(message, history):
        """处理用户消息"""
        # 将Gradio history格式转为内部格式
        internal_history = []
        for h in (history or []):
            if isinstance(h, dict):
                internal_history.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            elif isinstance(h, (list, tuple)) and len(h) >= 2:
                internal_history.append({"role": "user", "content": str(h[0])})
                internal_history.append({"role": "assistant", "content": str(h[1])})

        response = conv_agent.chat(message, internal_history)
        return response

    examples = [
        "推荐几部类似《盗梦空间》的科幻悬疑片",
        "最近想看轻松搞笑的喜剧片，有什么推荐？",
        "我喜欢诺兰导演的作品，推荐几部他的代表作",
        "推荐几部评分最高的动画电影",
        "有没有类似《肖申克的救赎》那种越狱题材的好电影？",
    ]

    gr.Markdown(
        """## 🎬 智能电影推荐助手

欢迎！我是你的AI电影管家 **小影**。你可以直接用自然语言告诉我你的观影偏好，
我会结合多种推荐算法为你精准推荐电影。

**试试这样说：**
- "推荐几部类似《星际穿越》的科幻片"
- "我想看评分高但不是太烧脑的悬疑片"
- "有没有好看的日本动画电影推荐？"
        """
    )

    chat = gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(height=500, bubble_full_width=False),
        textbox=gr.Textbox(
            placeholder="输入你想看的电影类型、风格、或参考电影...",
            container=False,
            scale=7,
        ),
        title="",
        description="",
        examples=examples,
        cache_examples=False,
        theme="soft",
    )

    return chat


def create_model_dashboard():
    """创建模型对比仪表盘Tab"""
    import gradio as gr

    gr.Markdown("## 📊 模型性能对比仪表盘")

    # 加载评估结果
    model_dir = Config.get_project_root() / "data/processed/models"
    figures_dir = Config.get_project_root() / "data/processed/figures"

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 核心指标对比")

            # Auto-detect which models are trained and load real metrics
            import json
            model_names = ["popularity", "svd", "ncf", "lightgcn", "content_bert", "ensemble"]
            metrics_data = {}
            for name in model_names:
                metrics_path = model_dir / f"{name}_metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics_data[name] = json.load(f)
                elif (model_dir / f"{name}_model.pkl").exists():
                    metrics_data[name] = {"ndcg@10": "已训练", "hit_rate@10": "已训练", "rmse": "已训练"}

            if not metrics_data:
                gr.Markdown("*尚未训练任何模型。请先运行 `python run_pipeline.py`*")
            else:
                rows = []
                for name in model_names:
                    if name in metrics_data:
                        m = metrics_data[name]
                        ndcg = f"{m['ndcg@10']:.4f}" if isinstance(m.get("ndcg@10"), (int, float)) else str(m.get("ndcg@10", "-"))
                        hr = f"{m['hit_rate@10']:.4f}" if isinstance(m.get("hit_rate@10"), (int, float)) else str(m.get("hit_rate@10", "-"))
                        rmse = f"{m['rmse']:.4f}" if isinstance(m.get("rmse"), (int, float)) else str(m.get("rmse", "-"))
                        rows.append([name, ndcg, hr, rmse])
                    else:
                        rows.append([name, "未训练", "未训练", "未训练"])

                df = pd.DataFrame(rows, columns=["Model", "NDCG@10", "Hit Rate@10", "RMSE"])
                best_name = next((n for n in model_names if n in metrics_data), "N/A")
                gr.DataFrame(df, label=f"模型指标 (已训练: {len(metrics_data)}/{len(model_names)})", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### 推荐性能可视化")
            comp_path = figures_dir / "model_comparison.png"
            if comp_path.exists():
                gr.Image(str(comp_path), label="模型对比柱状图")
            else:
                gr.Markdown(
                    "*运行完整流水线后将自动生成对比图表。*\n"
                    "```bash\npython run_pipeline.py\n```"
                )


def create_data_insights():
    """创建数据洞察Tab"""
    import gradio as gr

    gr.Markdown("## 📈 数据洞察")

    figures_dir = Config.get_project_root() / "data/processed/figures"
    processed_dir = Config.get_project_root() / "data/processed"

    with gr.Row():
        dist_path = figures_dir / "data_distribution.png"
        if dist_path.exists():
            gr.Image(str(dist_path), label="评分与用户活跃度分布")
        else:
            gr.Markdown(
                "*运行完整流水线后将自动生成数据分布图表。*"
            )

    # 数据摘要
    gr.Markdown("### 数据集摘要")

    ratings_path = processed_dir / "ratings.parquet"
    if ratings_path.exists():
        ratings = pd.read_parquet(ratings_path)

        with gr.Row():
            gr.Metric("评分总数", f"{len(ratings):,}")
            gr.Metric("用户数", f"{ratings['user_idx'].nunique():,}")
            gr.Metric("电影数", f"{ratings['movie_idx'].nunique():,}")

        with gr.Row():
            gr.Metric("平均评分", f"{ratings['rating'].mean():.2f}")
            gr.Metric("评分标准差", f"{ratings['rating'].std():.2f}")
            sparsity = 1 - len(ratings) / (
                ratings["user_idx"].nunique() * ratings["movie_idx"].nunique()
            )
            gr.Metric("矩阵稀疏度", f"{sparsity:.3%}")

    # 热门电影展示
    gr.Markdown("### 🔥 热门电影 Top 10")
    movies = load_data_safe()
    if len(movies) > 0:
        # 按评分排序
        rating_col = None
        for col in ["averageRating", "vote_average"]:
            if col in movies.columns:
                rating_col = col
                break

        if rating_col:
            top = movies.sort_values(rating_col, ascending=False).head(10)
            title_col = "title" if "title" in top.columns else "primaryTitle"
            genre_col = "genres" if "genres" in top.columns else None

            top_data = []
            for _, row in top.iterrows():
                top_data.append([
                    str(row.get(title_col, "")),
                    str(row.get("startYear", "")),
                    str(row.get(genre_col or "genres", "")),
                    f"{float(row[rating_col]):.1f}",
                ])
            gr.DataFrame(
                pd.DataFrame(top_data, columns=["电影名称", "年份", "类型", "评分"]),
                interactive=False,
            )
    else:
        gr.Markdown("*电影数据尚未加载。请先运行 `python run_pipeline.py`*")


def create_about():
    """关于页面Tab"""
    import gradio as gr

    gr.Markdown(
        """
## ℹ️ 关于本项目

### 🎬 基于多智能体协同的大规模混合数据电影推荐系统

本项目是 **"人工智能与大数据辅助决策"** 课程项目，探索了AI Agent在推荐系统全流程中的应用。

---

### 🏗️ 技术架构

| 层级 | 技术 |
|------|------|
| 数据处理 | Pandas, NumPy, SciPy |
| 传统推荐 | SVD++ (Surprise) |
| 深度学习 | PyTorch, NCF, LightGCN |
| 文本编码 | Sentence-BERT |
| LLM对话 | OpenAI GPT API |
| Web界面 | Gradio |
| 可视化 | Matplotlib, Plotly |

---

### 🤖 多Agent协同

- **DataAgent**: 多源数据下载、清洗、融合
- **ModelingAgent**: 推荐模型训练、评估、选优
- **AnalysisAgent**: 结果解读与可视化
- **ConversationAgent**: 自然语言对话推荐

---

### 📊 数据源

- **MovieLens 25M**: 2500万条用户评分
- **IMDB Datasets**: 电影元信息与演职人员
- **TMDB API**: 电影概述与海报

---

### 🚀 快速开始

```bash
# 1. 设置API Key
export OPENAI_API_KEY="your-key"
export TMDB_API_KEY="your-key"

# 2. 运行完整流水线
python run_pipeline.py

# 3. 启动Web界面
python main.py ui
```

---
*© 2026 人工智能与大数据辅助决策课程项目*
        """
    )


def launch_ui():
    """启动Gradio Web界面"""
    import gradio as gr

    ui_config = Config.get("ui", {})

    # 自定义CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chatbot {
        border-radius: 10px !important;
    }
    """

    with gr.Blocks(
        title=ui_config.get("title", "🎬 智能电影推荐助手"),
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:
        gr.Markdown(
            """# 🎬 基于多智能体协同的电影推荐系统
            ### AI与大数��辅助决策课程项目""",
            elem_id="main-title",
        )

        with gr.Tabs():
            with gr.TabItem("💬 对话推荐", id="chat"):
                create_chat_interface()

            with gr.TabItem("📊 模型对比", id="dashboard"):
                create_model_dashboard()

            with gr.TabItem("📈 数据洞察", id="insights"):
                create_data_insights()

            with gr.TabItem("ℹ️ 关于", id="about"):
                create_about()

    logger.info(f"启动 UI: http://{ui_config.get('host', '127.0.0.1')}:{ui_config.get('port', 7860)}")
    app.launch(
        server_name=ui_config.get("host", "127.0.0.1"),
        server_port=ui_config.get("port", 7860),
        share=ui_config.get("share", False),
    )


if __name__ == "__main__":
    launch_ui()
