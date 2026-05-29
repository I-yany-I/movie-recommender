"""Agent基类 — 定义统一的Agent接口和生命周期管理"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass
class AgentResult:
    """Agent执行结果"""
    agent_name: str
    status: AgentStatus
    data: Any = None
    report: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class BaseAgent(ABC):
    """Agent抽象基类 — 所有专业Agent继承此类"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self._start_time: float = 0.0
        self.logger = logging.getLogger(f"agent.{name}")

    def run(self, task_spec: Dict[str, Any] = None) -> AgentResult:
        """执行Agent任务的统一入口"""
        self.status = AgentStatus.RUNNING
        self._start_time = time.time()
        self.logger.info(f"[{self.name}] 开始执行任务...")
        try:
            data = self._execute(task_spec or {})
            report = self.report()
            elapsed = time.time() - self._start_time
            self.status = AgentStatus.DONE
            self.logger.info(
                f"[{self.name}] 任务完成，耗时 {elapsed:.1f}s"
            )
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.DONE,
                data=data,
                report=report,
                metadata={"elapsed_seconds": elapsed},
                elapsed_seconds=elapsed,
            )
        except Exception as e:
            self.status = AgentStatus.ERROR
            elapsed = time.time() - self._start_time
            self.logger.error(f"[{self.name}] 任务失败: {e}", exc_info=True)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.ERROR,
                error=str(e),
                metadata={"elapsed_seconds": elapsed},
                elapsed_seconds=elapsed,
            )

    @abstractmethod
    def _execute(self, task_spec: Dict[str, Any]) -> Any:
        """子类实现：核心执行逻辑"""
        ...

    @abstractmethod
    def report(self) -> str:
        """子类实现：生成人类可读的执行报告"""
        ...

    def __repr__(self):
        return f"<{self.name} status={self.status.value}>"
