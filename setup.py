#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键安装脚本 — 电影推荐系统

用法:
  python setup.py --all        # 完整安装：下载数据+预处理+训练+启动UI
  python setup.py --quick      # 快速体验：下载数据+预处理，跳过训练
  python setup.py --train      # 仅训练模型
  python setup.py --ui         # 仅启动Web界面
  python setup.py --check      # 检查环境和依赖

首次运行建议: python setup.py --all
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║  🎬 多智能体协同电影推荐系统 — 一键安装              ║
║  AI & Big Data Assisted Decision Making Project      ║
╚══════════════════════════════════════════════════════╝
""")


def run_cmd(cmd: str, description: str = "", check: bool = True):
    """运行命令并显示进度"""
    if description:
        print(f"\n▶ {description}")
        print(f"  $ {cmd}")
    print("-" * 50)

    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        capture_output=False, text=True
    )

    if check and result.returncode != 0:
        print(f"\n❌ 失败 (exit code {result.returncode})")
        return False
    return True


def check_environment():
    """检查 Python 环境和依赖"""
    print("\n🔍 检查环境...")

    # Python version
    py_ver = sys.version_info
    if py_ver < (3, 9):
        print(f"❌ Python 版本: {py_ver.major}.{py_ver.minor} (需要 >= 3.9)")
        return False
    print(f"✅ Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    # Check pip
    try:
        import pip
        print(f"✅ pip {pip.__version__}")
    except ImportError:
        print("❌ pip 未安装")
        return False

    # Check key packages
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "gradio": "gradio",
        "yaml": "pyyaml",
    }
    for module, pkg_name in packages.items():
        try:
            __import__(module)
            print(f"✅ {pkg_name}")
        except ImportError:
            print(f"⚠️  {pkg_name} 未安装 (稍后将自动安装)")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用 (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("ℹ️  CUDA 不可用，将使用 CPU 训练（较慢但可用）")
    except Exception:
        pass

    # Check LLM API key
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✅ LLM API Key 已配置 ({'DeepSeek' if 'DEEPSEEK' in os.environ else 'OpenAI'})")
    else:
        print("ℹ️  未设置 API Key — 对话推荐将使用离线模式")
        print("   设置方法: export DEEPSEEK_API_KEY=your-key")

    # Check disk space
    try:
        import shutil
        usage = shutil.disk_usage(PROJECT_ROOT)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 5:
            print(f"⚠️  可用磁盘空间: {free_gb:.1f}GB (建议 > 10GB)")
        else:
            print(f"✅ 可用磁盘空间: {free_gb:.1f}GB")
    except Exception:
        pass

    return True


def install_dependencies():
    """安装 Python 依赖"""
    print("\n📦 安装 Python 依赖...")
    return run_cmd(
        f"{sys.executable} -m pip install -r requirements.txt -q",
        "安装依赖包"
    )


def download_data():
    """下载数据集"""
    print("\n📥 下载数据...")

    # Check if data already exists
    ml_path = PROJECT_ROOT / "data/raw/ml-25m/ratings.csv"
    imdb_path = PROJECT_ROOT / "data/raw/imdb/title.basics.tsv"

    if ml_path.exists() and imdb_path.exists():
        print("✅ 数据已存在，跳过下载")
        return True

    return run_cmd(
        f"{sys.executable} main.py download",
        "下载 MovieLens 25M + IMDB 数据集 (~2GB)"
    )


def preprocess_data():
    """预处理数据"""
    print("\n🧹 预处理数据...")

    ratings_path = PROJECT_ROOT / "data/processed/ratings.parquet"
    if ratings_path.exists():
        print("✅ 预处理数据已存在，跳过")
        return True

    return run_cmd(
        f"{sys.executable} main.py preprocess",
        "数据清洗、过滤、融合、特征工程"
    )


def train_models(models: str = "all"):
    """训练推荐模型"""
    print(f"\n🤖 训练模型: {models}...")

    # Check if all models already trained
    model_dir = PROJECT_ROOT / "data/processed/models"
    required = ["popularity_model.pkl", "svd_model.pkl", "lightgcn_model.pkl",
                 "ncf_model.pkl", "content_bert_model.pkl", "ensemble_model.pkl"]

    if all((model_dir / f).exists() for f in required):
        print("✅ 所有模型已训练，跳过")
        print("   如需重新训练，请删除 data/processed/models/ 目录")
        return True

    if models == "quick":
        # Only train fast models
        for model in ["popularity", "svd"]:
            run_cmd(
                f"{sys.executable} main.py train --model {model}",
                f"训练 {model} 模型"
            )
        return True

    return run_cmd(
        f"{sys.executable} main.py train --model {models}",
        f"训练推荐模型 ({models})"
    )


def run_pipeline():
    """运行完整 Agent 流水线"""
    print("\n🔄 运行 Agent 协同流水线...")
    return run_cmd(
        f"{sys.executable} main.py pipeline",
        "Data → Modeling → Analysis Agent 协同执行"
    )


def launch_ui():
    """启动 Web UI"""
    print("\n🖥️  启动 Web 界面...")
    print("  打开浏览器访问: http://127.0.0.1:7860")
    print("  按 Ctrl+C 停止服务器\n")
    return run_cmd(
        f"{sys.executable} main.py ui",
        "启动 Gradio Web UI",
        check=False
    )


def run_tests():
    """运行单元测试"""
    print("\n🧪 运行单元测试...")
    return run_cmd(
        f"{sys.executable} -m pytest tests/ -v",
        "测试评估指标、模型、配置"
    )


def show_status():
    """显示当前项目状态"""
    print("\n📊 项目状态:")
    print("-" * 40)

    checks = [
        ("Python 依赖", lambda: True),
        ("原始数据 (MovieLens)",
         lambda: (PROJECT_ROOT / "data/raw/ml-25m/ratings.csv").exists()),
        ("原始数据 (IMDB)",
         lambda: (PROJECT_ROOT / "data/raw/imdb/title.basics.tsv").exists()),
        ("预处理数据",
         lambda: (PROJECT_ROOT / "data/processed/ratings.parquet").exists()),
        ("Popularity 模型",
         lambda: (PROJECT_ROOT / "data/processed/models/popularity_model.pkl").exists()),
        ("SVD 模型",
         lambda: (PROJECT_ROOT / "data/processed/models/svd_model.pkl").exists()),
        ("NCF 模型",
         lambda: (PROJECT_ROOT / "data/processed/models/ncf_model.pkl").exists()),
        ("LightGCN 模型",
         lambda: (PROJECT_ROOT / "data/processed/models/lightgcn_model.pkl").exists()),
        ("Content-BERT 模型",
         lambda: (PROJECT_ROOT / "data/processed/models/content_bert_model.pkl").exists()),
        ("Ensemble 模型",
         lambda: (PROJECT_ROOT / "data/processed/models/ensemble_model.pkl").exists()),
        ("BERT 嵌入",
         lambda: (PROJECT_ROOT / "data/processed/embeddings/movie_bert_embeddings.npy").exists()),
        ("分析报告",
         lambda: (PROJECT_ROOT / "data/processed/analysis_report.md").exists()),
        ("可视化图表",
         lambda: (PROJECT_ROOT / "data/processed/figures/model_comparison.png").exists()),
    ]

    all_ok = True
    for name, check_fn in checks:
        try:
            result = check_fn()
            if result:
                print(f"  ✅ {name}")
            else:
                print(f"  ⬜ {name}")
                all_ok = False
        except Exception:
            print(f"  ❓ {name}")

    print("-" * 40)
    if all_ok:
        print("  🎉 项目完整就绪！运行 python main.py ui 启动系统")
    else:
        print("  💡 运行 python setup.py --all 完成安装")

    # Show model metrics if available
    metrics_dir = PROJECT_ROOT / "data/processed/models"
    metrics_files = list(metrics_dir.glob("*_metrics.json"))
    if metrics_files:
        print(f"\n📈 已评估模型: {len(metrics_files)} 个")


def main():
    parser = argparse.ArgumentParser(
        description="🎬 多智能体协同电影推荐系统 — 一键安装脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python setup.py --all       # 完整安装
  python setup.py --quick     # 快速体验 (跳过训练)
  python setup.py --check     # 检查环境
  python setup.py --ui        # 启动 Web 界面
  python setup.py --status    # 查看当前状态
        """
    )
    parser.add_argument("--all", action="store_true", help="完整安装流程")
    parser.add_argument("--quick", action="store_true", help="快速体验 (下载+预处理，跳过训练)")
    parser.add_argument("--check", action="store_true", help="检查环境和依赖")
    parser.add_argument("--train", action="store_true", help="仅训练模型")
    parser.add_argument("--train-quick", action="store_true", help="仅训练快速模型 (popularity+svd)")
    parser.add_argument("--ui", action="store_true", help="启动 Web UI")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--status", action="store_true", help="显示项目状态")

    args = parser.parse_args()

    print_banner()

    # --status
    if args.status:
        show_status()
        return

    # --check
    if args.check:
        check_environment()
        return

    # --ui (standalone)
    if args.ui and not args.all:
        launch_ui()
        return

    # --test
    if args.test:
        run_tests()
        return

    # --train / --train-quick
    if args.train:
        check_environment()
        install_dependencies()
        if preprocess_data():
            train_models("all")
            run_pipeline()
        return

    if args.train_quick:
        check_environment()
        install_dependencies()
        if preprocess_data():
            train_models("quick")
            run_pipeline()
        return

    # --quick
    if args.quick:
        check_environment()
        install_dependencies()
        download_data()
        preprocess_data()
        run_pipeline()
        launch_ui()
        return

    # --all (default when no args)
    if args.all or len(sys.argv) == 1:
        check_environment()
        install_dependencies()
        download_data()
        preprocess_data()
        train_models("all")
        run_pipeline()
        run_tests()
        show_status()
        print("\n🎉 安装完成！")
        print("运行 python main.py ui 启动 Web 界面")
        return

    # No valid args
    parser.print_help()


if __name__ == "__main__":
    main()
