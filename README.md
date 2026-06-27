# 🎬 基于多智能体协同的大规模混合数据电影推荐系统

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 探索 AI Agent 在推荐系统全流程中的应用 —— 从数据采集到对话式推荐，实现用自然语言获取个性化电影推荐。

---

## 🎯 项目简介

本系统是一个**多智能体协同的电影推荐系统**，创新地将 4 个 AI Agent（数据智能体、建模智能体、分析智能体、对话智能体）协同应用于推荐系统的完整生命周期。用户通过**自然语言对话**即可获取个性化电影推荐，系统会为每一条推荐提供**多维度推荐理由**。

### ✨ 核心特色

- 🤖 **四 Agent 协同架构** — Data → Modeling → Analysis → Conversation
- 💬 **对话式推荐** — 用自然语言描述偏好，像聊天一样获取推荐
- 🔗 **三源数据融合** — MovieLens 25M + IMDB + TMDB
- 🧠 **混合推荐引擎** — SVD++ / NCF / LightGCN / BERT / Ensemble
- 📊 **全链路可解释** — 每条推荐附多维度推荐理由
- 🎨 **CinemaScope 暗金影院 Web** — FastAPI + Jinja2 构建的沉浸式电影探索体验
- 🖥️ **Gradio 仪表盘** — 数据分析与模型对比的可视化界面

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│              表现层: CinemaScope Web + Gradio 仪表盘           │
│     🎨 暗金影院 (FastAPI)  │  💬对话推荐  │  📊模型对比       │
├──────────────────────────────────────────────────────────────┤
│          服务层: MovieService / ChatService / TMDBService     │
├──────────────────────────────────────────────────────────────┤
│          对话智能体 (DeepSeek LLM + Function Calling)          │
│              任务分解 → 分发 → 结果聚合                        │
├────────────────┬──────────────────┬──────────────────────────┤
│   Data Agent   │  Modeling Agent  │  Analysis Agent          │
│   数据采集清洗   │  模型训练评估     │  结果解读可视化           │
├────────────────┴──────────────────┴──────────────────────────┤
│       模型层: SVD++ | NCF | LightGCN | BERT | Ensemble       │
├──────────────────────────────────────────────────────────────┤
│        数据层: MovieLens CSV | IMDB TSV | TMDB JSON           │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- CUDA (可选，CPU 也可运行)
- 磁盘空间：~10GB（用于数据和模型）

### 方式一：一键安装运行

```bash
# 1. 克隆项目
git clone https://github.com/I-yany-I/movie-recommender.git
cd movie-recommender

# 2. 一键安装 + 下载数据 + 训练模型 + 启动UI
python setup.py --all
```

### 方式二：分步操作

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据（MovieLens + IMDB，约 2GB）
python main.py download

# 3. 预处理数据
python main.py preprocess

# 4. 训练模型（可选 all / svd / ncf / lightgcn / content_bert）
python main.py train --model all

# 5. 启动 Web 界面（二选一）
python main.py serve              # CinemaScope 暗金影院 (http://127.0.0.1:8000)
python main.py ui                 # Gradio 数据分析仪表盘 (http://127.0.0.1:7860)
```

### 方式三：快速体验（无需训练）

```bash
# 仅安装依赖 + 下载数据 + 预处理，跳过训练
python setup.py --quick

# 启动 UI（使用 Popularity 基线模型 + BERT 内容推荐）
python main.py ui
```

---

## 🎨 CinemaScope 暗金影院

启动 `python main.py serve` 后访问 http://127.0.0.1:8000 体验：

| 页面 | 路径 | 功能 |
|------|------|------|
| 首页 | `/` | 热门推荐 + 类型探索 |
| 搜索 | `/search?q=Inception` | 中英文模糊搜索 |
| 电影详情 | `/movie/{id}` | 海报/概述/演职员 + **YouTube 预告片嵌入播放** + 流媒体提供商 |
| AI 对话 | `/chat` | 自然语言对话推荐 + TMDB 富化电影卡片 |
| 关于 | `/about` | 项目介绍与技术栈 |
| API 文档 | `/docs` | Swagger UI — 8 个 REST 端点 |

**设计**: 暗金影院风 (`#0a0a0a` 深黑 + `#c9a96e` 金色 + Playfair Display 字体)

---

## 📊 模型性能

| 模型 | RMSE ↓ | NDCG@10 ↑ | HR@10 ↑ | 说明 |
|------|--------|-----------|---------|------|
| Popularity (基线) | 1.334 | 0.060 | 0.132 | 热门推荐基线 |
| SVD++ | 0.964 | 0.042 | 0.094 | 矩阵分解 |
| NCF | 0.985 | 0.052 | 0.112 | 神经协同过滤 |
| LightGCN | 1.158 | 0.025 | 0.052 | 图卷积推荐 |
| Content-BERT | 1.074 | 0.038 | 0.084 | 文本内容推荐 |
| **Ensemble** | **0.872** | **0.071** | **0.151** | 多模型加权融合 |

*测试条件：MovieLens-25M, 162K 用户, 18K 电影, 时间序列切分*

---

## 📁 项目结构

```
movie-recommender/
├── main.py                 # CLI 入口 (download/preprocess/train/ui/pipeline)
├── setup.py                # 一键安装脚本
├── run_pipeline.py         # 完整流水线
├── config.yaml             # 全局配置
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
├── src/
│   ├── agents/             # 四 Agent 实现
│   │   ├── base_agent.py       # Agent 抽象基类
│   │   ├── data_agent.py       # 数据智能体
│   │   ├── modeling_agent.py   # 建模智能体
│   │   ├── analysis_agent.py   # 分析智能体
│   │   └── conversation_agent.py # 对话智能体 (LLM)
│   ├── models/             # 推荐模型实现
│   │   ├── base.py             # 模型基类
│   │   ├── popularity.py       # 热门推荐
│   │   ├── svd_model.py        # SVD++ 矩阵分解
│   │   ├── ncf_model.py        # Neural CF
│   │   ├── lightgcn.py         # LightGCN
│   │   ├── content_bert.py     # BERT 文本推荐
│   │   └── ensemble.py         # 集成模型
│   ├── data/               # 数据处理流水线
│   │   ├── downloader.py       # 数据下载
│   │   ├── preprocessor.py     # 数据预处理
│   │   ├── merger.py           # 多源融合
│   │   └── feature_engineer.py # 特征工程
│   ├── evaluation/         # 评估指标
│   │   └── metrics.py
│   ├── api/                # FastAPI 后端 (CinemaScope)
│   │   ├── main.py             # 应用工厂 + lifespan
│   │   ├── schemas.py          # Pydantic 模型
│   │   └── routes/             # 电影/对话/推荐/页面路由
│   ├── services/            # 业务服务层
│   │   ├── tmdb_service.py     # TMDB API 封装 (16K+ 电影缓存)
│   │   ├── movie_service.py    # 电影搜索/详情/相似/热门
│   │   ├── chat_service.py     # 对话推荐 + 结构化卡片
│   │   └── recommendation_service.py
│   ├── web/                 # 暗金影院前端
│   │   ├── templates/          # 6 个 Jinja2 模板
│   │   └── static/             # CSS + JS + 图片
│   ├── ui/                 # Gradio 仪表盘
│   │   └── app.py
│   ├── orchestrator.py     # Agent 编排器
│   └── config.py           # 配置管理器
├── tests/                  # 单元测试 (32个)
│   ├── test_metrics.py
│   ├── test_popularity.py
│   └── test_config.py
└── data/
    ├── raw/                # 原始数据 (需下载)
    ├── processed/          # 处理后数据 + 模型 + 结果
    │   ├── models/         # 训练好的模型
    │   ├── figures/        # 可视化图表
    │   ├── embeddings/     # BERT 嵌入
    │   └── analysis_report.md
    └── .gitkeep
```

---

## 🗣️ 使用示例

启动 UI 后，在对话推荐 Tab 中输入：

| 用户输入 | 系统行为 |
|---------|---------|
| "推荐几部类似《盗梦空间》的科幻悬疑片" | 搜索+内容相似推荐 |
| "最近想看轻松搞笑的喜剧片" | 类型+偏好推荐 |
| "我喜欢诺兰导演的作品" | 导演筛选+评分排序 |
| "推荐几部评分最高的动画电影" | 类型过滤+热门推荐 |
| "有没有类似《肖申克的救赎》的电影？" | BERT 语义相似度推荐 |

---

## 🔧 配置说明

编辑 `config.yaml` 可调整：

- **数据过滤阈值**：`preprocessing.min_user_ratings` / `min_movie_ratings`
- **模型超参数**：各模型的学习率、嵌入维度、层数等
- **LLM 配置**：`llm.provider` / `llm.model` / `llm.base_url`
- **评估指标**：`evaluation.metrics` / `evaluation.top_k`

LLM API Key 通过环境变量设置：

```bash
# Windows
set DEEPSEEK_API_KEY=your-api-key

# Linux/Mac
export DEEPSEEK_API_KEY=your-api-key
```

---

## 🧪 运行测试

```bash
pytest tests/ -v
# 32 passed ✅
```

---

## 📚 数据来源

| 数据集 | 规模 | 链接 |
|--------|------|------|
| MovieLens 25M | 2500万评分 / 6.2万电影 / 16.2万用户 | [GroupLens](https://grouplens.org/datasets/movielens/25m/) |
| IMDB Datasets | 100万+电影元信息 | [IMDB](https://datasets.imdbws.com/) |
| TMDB API | 80万+电影概述 | [TMDB](https://www.themoviedb.org/) |

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE)

---

*技术栈：Python · PyTorch · Transformers · FAISS · FastAPI · Gradio · DeepSeek API*
