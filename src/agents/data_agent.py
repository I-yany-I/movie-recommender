"""数据Agent — 自动化数据下载、清洗、融合、特征工程的完整流水线"""
import logging
from typing import Any, Dict

from .base_agent import BaseAgent, AgentStatus

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """数据智能体 — 负责数据获取、清洗、多源融合和数据质量保障"""

    def __init__(self):
        super().__init__(
            name="DataAgent",
            description="多源数据采集、清洗、融合与数据质量报告专家Agent",
        )

    def _execute(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据处理流水线"""
        results = {}

        # Phase 1: 下载数据
        self.logger.info("📥 Phase 1/4: 下载数据源...")
        from ..data.downloader import DataDownloader
        dl = DataDownloader()
        dl.download_all()
        results["download"] = "done"

        # Phase 2: 预处理
        self.logger.info("🧹 Phase 2/4: 数据清洗与预处理...")
        from ..data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        stats = preprocessor.run()
        results["preprocessing"] = stats

        # Phase 3: 多源融合
        self.logger.info("🔗 Phase 3/4: 多源数据融合...")
        from ..data.merger import DataMerger
        merger = DataMerger()
        merger.run()
        results["merge"] = "done"

        # Phase 4: 特征工程
        self.logger.info("🔧 Phase 4/4: 特征工程...")
        from ..data.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        fe_result = fe.run()
        results["features"] = {k: "done" if v is not None else "skipped" for k, v in fe_result.items()}

        return results

    def report(self) -> str:
        """生成数据质量报告"""
        from pathlib import Path
        from ..config import Config

        project_root = Config.get_project_root()
        processed_dir = project_root / "data/processed"

        report_lines = [
            "=" * 50,
            "  📊 数据Agent — 执行报告",
            "=" * 50,
            "",
            "【数据源】",
            "  - MovieLens 25M: 用户评分数据",
            "  - IMDB Datasets: 电影元信息（类型、导演、演员）",
            "  - TMDB API: 电影概述与海报信息",
            "",
            "【数据质量】",
            "  - 用户过滤: 至少20次评分",
            "  - 电影过滤: 至少被20次评分",
            "  - 数据划分: 时间序列切分（训练/验证/测试）",
            "  - 缺失值处理: 自动检测并填充",
            "",
        ]

        # 检查处理后的数据
        for fname, desc in [
            ("ratings.parquet", "评分数据"),
            ("movies.parquet", "电影基础信息"),
            ("unified_movies.parquet", "融合后的统一电影表"),
            ("kg_triples.json", "知识图谱三元组"),
        ]:
            path = processed_dir / fname
            if path.exists():
                size_kb = path.stat().st_size / 1024
                report_lines.append(f"  ✅ {desc}: {fname} ({size_kb:.1f} KB)")
            else:
                report_lines.append(f"  ❌ {desc}: {fname} (未找到)")

        # BERT嵌入
        emb_path = processed_dir / "embeddings" / "movie_bert_embeddings.npy"
        if emb_path.exists():
            import numpy as np
            emb = np.load(emb_path)
            report_lines.append(f"  ✅ BERT文本嵌入: shape={emb.shape}")

        report_lines.append("")
        report_lines.append("【特征工程】")
        report_lines.append("  - BERT文本嵌入维度: 384")
        report_lines.append("  - 知识图谱三元组: 电影-类型-导演-演员")
        report_lines.append("  - 类型多热编码: 可用于内容过滤")
        report_lines.append("")
        report_lines.append("=" * 50)

        return "\n".join(report_lines)
