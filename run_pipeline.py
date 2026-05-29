"""一键运行完整流水线：数据→建模→分析→UI"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.orchestrator import Orchestrator


def main():
    print("=" * 60)
    print("  🎬 多智能体协同电影推荐系统 — 完整流水线")
    print("=" * 60)

    orchestrator = Orchestrator()
    results = orchestrator.run_full_pipeline()

    print("\n" + "=" * 60)
    print("  流水线执行完毕！")
    print("=" * 60)

    for r in results:
        status_icon = "✅" if r.status.value == "DONE" else "❌"
        print(f"  {status_icon} {r.agent_name}: {r.status.value} ({r.elapsed_seconds:.1f}s)")

    print("\n启动 Web UI: python main.py ui")
    print("=" * 60)


if __name__ == "__main__":
    main()
