"""项目入口 — 提供CLI命令用于数据准备、模型训练、UI启动"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config


def cmd_download(args):
    """下载所有数据源"""
    from src.data.downloader import DataDownloader
    dl = DataDownloader()
    dl.download_all()


def cmd_preprocess(args):
    """运行完整数据预处理流水线"""
    from src.data.preprocessor import DataPreprocessor
    from src.data.merger import DataMerger
    preprocessor = DataPreprocessor()
    preprocessor.run()
    merger = DataMerger()
    merger.run()


def _prepare_data():
    """Load processed data and split into train/val/test sets."""
    import pandas as pd
    from src.data.preprocessor import DataPreprocessor
    from src.config import Config

    preprocessor = DataPreprocessor()
    processed_dir = Config.get_project_root() / "data/processed"
    ratings_path = processed_dir / "ratings.parquet"

    if ratings_path.exists():
        # Fast path: load preprocessed parquet
        preprocessor.filtered_ratings = pd.read_parquet(ratings_path)
        preprocessor.n_users = int(preprocessor.filtered_ratings["user_idx"].max()) + 1
        preprocessor.n_movies = int(preprocessor.filtered_ratings["movie_idx"].max()) + 1
        preprocessor.n_ratings = len(preprocessor.filtered_ratings)
    else:
        preprocessor.run()

    return preprocessor.get_train_test_split()


def cmd_train(args):
    """训练指定模型"""
    from src.models.popularity import PopularityModel
    from src.models.svd_model import SVDModel
    from src.models.ncf_model import NCFModel
    from src.models.lightgcn import LightGCNModel
    from src.models.content_bert import ContentBERTModel

    model_name = args.model
    models = {
        "popularity": PopularityModel,
        "svd": SVDModel,
        "ncf": NCFModel,
        "lightgcn": LightGCNModel,
        "content_bert": ContentBERTModel,
    }

    train_df, val_df, test_df = _prepare_data()

    if model_name == "all":
        for name, cls in models.items():
            print(f"\n{'='*50}\n  Training {name}\n{'='*50}")
            m = cls()
            m.train(train_df, val_df)
            m.evaluate(test_df)
            m.save()
    elif model_name in models:
        m = models[model_name]()
        m.train(train_df, val_df)
        m.evaluate(test_df)
        m.save()
    else:
        print(f"Unknown model: {model_name}. Available: {list(models.keys())} + 'all'")


def cmd_pipeline(args):
    """运行完整的Agent流水线"""
    from src.orchestrator import Orchestrator
    orch = Orchestrator()
    orch.run_full_pipeline()


def cmd_ui(args):
    """启动Gradio Web界面"""
    from src.ui.app import launch_ui
    launch_ui()


def main():
    parser = argparse.ArgumentParser(
        description="🎬 多智能体协同电影推荐系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py download            # 下载所有数据
  python main.py preprocess          # 预处理和融合数据
  python main.py train --model svd   # 训练单个模型
  python main.py train --model all   # 训练所有模型
  python main.py pipeline            # 运行完整Agent流水线
  python main.py ui                  # 启动Web界面
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("download", help="下载MovieLens + IMDB数据")
    subparsers.add_parser("preprocess", help="数据预处理与融合")
    train_parser = subparsers.add_parser("train", help="训练推荐模型")
    train_parser.add_argument("--model", default="all", help="模型名称或'all'")
    subparsers.add_parser("pipeline", help="运行完整Agent流水线")
    subparsers.add_parser("ui", help="启动Gradio Web界面")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    commands = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "pipeline": cmd_pipeline,
        "ui": cmd_ui,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
