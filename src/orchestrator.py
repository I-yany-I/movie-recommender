"""多Agent编排器 — 协调Data/Modeling/Analysis Agent的协同执行"""
import logging
import sys
from pathlib import Path
from typing import List

from .config import Config
from .agents.base_agent import AgentResult
from .agents.data_agent import DataAgent
from .agents.modeling_agent import ModelingAgent
from .agents.analysis_agent import AnalysisAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """多Agent协同编排器 — 按流水线顺序调度各Agent"""

    def __init__(self, skip_data: bool = False, skip_training: bool = False):
        self.skip_data = skip_data
        self.skip_training = skip_training
        self.results: List[AgentResult] = []

        # 初始化日志
        logging.basicConfig(
            level=getattr(logging, Config.get("logging.level", "INFO")),
            format=Config.get("logging.format"),
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    Config.get_project_root() / "pipeline.log",
                    encoding="utf-8",
                ),
            ],
        )

    def run_full_pipeline(self) -> List[AgentResult]:
        """运行完整的Agent协作流水线"""
        logger.info("=" * 60)
        logger.info("  🎬 多智能体协同流水线启动")
        logger.info("=" * 60)

        # Phase 1: Data Agent
        if not self.skip_data:
            logger.info("\n▶ Phase 1/3: Data Agent 启动")
            data_agent = DataAgent()
            result = data_agent.run()
            self.results.append(result)
            if result.status.value == "ERROR":
                logger.error("Data Agent 失败，流水线终止")
                return self.results
            logger.info(result.report())
        else:
            logger.info("\n▶ Phase 1/3: 跳过 (数据已就绪)")

        # Phase 2: Modeling Agent
        if not self.skip_training:
            logger.info("\n▶ Phase 2/3: Modeling Agent 启动")
            modeling_agent = ModelingAgent()
            # 传递需要训练的模型列表
            task_spec = {
                "models": ["popularity", "svd", "ncf", "lightgcn", "content_bert"],
            }
            result = modeling_agent.run(task_spec)
            self.results.append(result)
            if result.status.value == "ERROR":
                logger.error("Modeling Agent 失败")
            logger.info(result.report())
        else:
            logger.info("\n▶ Phase 2/3: 跳过 (模型已训练)")

        # Phase 3: Analysis Agent
        logger.info("\n▶ Phase 3/3: Analysis Agent 启动")
        analysis_agent = AnalysisAgent()
        result = analysis_agent.run({"case_user_id": 42})
        self.results.append(result)
        logger.info(result.report())

        self._print_summary()
        return self.results

    def run_interactive(self, user_query: str) -> str:
        """对话模式：接收用户自然语言查询，调度Agent返回推荐"""
        from .agents.conversation_agent import ConversationAgent

        conv_agent = ConversationAgent()
        return conv_agent.chat(user_query)

    def _print_summary(self):
        """打印流水线执行摘要"""
        print("\n" + "=" * 60)
        print("  📋 流水线执行摘要")
        print("=" * 60)
        total_time = 0
        for r in self.results:
            status = "✅" if r.status.value == "DONE" else "❌"
            print(
                f"  {status} {r.agent_name:<20s} "
                f"{r.status.value:<6s} "
                f"({r.elapsed_seconds:.1f}s)"
            )
            total_time += r.elapsed_seconds
        print(f"\n  总耗时: {total_time:.1f}s")
        print("=" * 60)
