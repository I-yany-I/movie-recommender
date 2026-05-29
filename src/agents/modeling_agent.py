"""建模Agent — 自动训练、评估和对比多个推荐模型"""
import json
import logging
import time
from typing import Any, Dict, List

from .base_agent import BaseAgent
from ..config import Config
from ..evaluation.metrics import build_test_dict, evaluate_recommendations

logger = logging.getLogger(__name__)


class ModelingAgent(BaseAgent):
    """建模智能体 — 模型训练调度、评估对比与模型选优"""

    def __init__(self):
        super().__init__(
            name="ModelingAgent",
            description="推荐模型训练、自动评估与模型选优专家Agent",
        )
        self.model_results: Dict[str, Dict] = {}

    def _execute(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """执行模型训练与评估流水线"""
        self.logger.info("=" * 50)
        self.logger.info("开始模型训练与评估...")
        self.logger.info("=" * 50)

        # 加载预处理数据
        from ..data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.run()
        train_df, val_df, test_df = preprocessor.get_train_test_split()

        # 构建测试集的ground truth
        test_dict = build_test_dict(test_df)

        # 训练并评估每个模型
        models_to_train = task_spec.get("models", [
            "popularity", "svd", "ncf", "lightgcn", "content_bert",
        ])

        for model_name in models_to_train:
            try:
                self.logger.info(f"\n{'='*40}\n  训练 {model_name}\n{'='*40}")
                result = self._train_and_evaluate(model_name, train_df, val_df, test_df, test_dict)
                self.model_results[model_name] = result
            except Exception as e:
                self.logger.error(f"模型 {model_name} 训练失败: {e}", exc_info=True)
                self.model_results[model_name] = {"error": str(e)}

        # 选择最佳模型
        best_model = self._select_best_model()
        self.model_results["_best"] = best_model

        # 训练集成模型
        try:
            self.logger.info(f"\n{'='*40}\n  训练 Ensemble 集成模型\n{'='*40}")
            self._train_ensemble(train_df, val_df, test_df, test_dict)
        except Exception as e:
            self.logger.error(f"集成模型训练失败: {e}")

        return self.model_results

    def _train_and_evaluate(self, model_name: str, train_df, val_df, test_df,
                            test_dict: Dict) -> Dict[str, Any]:
        """训练单个模型并评估"""
        t0 = time.time()

        if model_name == "popularity":
            from ..models.popularity import PopularityModel
            model = PopularityModel()
        elif model_name == "svd":
            from ..models.svd_model import SVDModel
            model = SVDModel()
        elif model_name == "ncf":
            from ..models.ncf_model import NCFModel
            model = NCFModel()
        elif model_name == "lightgcn":
            from ..models.lightgcn import LightGCNModel
            model = LightGCNModel()
        elif model_name == "content_bert":
            from ..models.content_bert import ContentBERTModel
            model = ContentBERTModel()
        else:
            raise ValueError(f"未知模型: {model_name}")

        model.train(train_df, val_df)
        metrics = model.evaluate(test_df)
        model.save()

        # 生成推荐评估
        predictions = {}
        for uid in list(test_dict.keys())[:500]:  # 采样500用户加速
            predictions[uid] = model.recommend(int(uid), top_k=10)
        ranking_metrics = evaluate_recommendations(predictions, test_dict, k=10)

        elapsed = time.time() - t0
        result = {
            "model": model_name,
            "metrics": metrics,
            "ranking_metrics": ranking_metrics,
            "training_time": elapsed,
        }

        self.logger.info(f"[{model_name}] 指标: {metrics}")
        return result

    def _train_ensemble(self, train_df, val_df, test_df, test_dict):
        """训练集成模型"""
        from ..models.ensemble import EnsembleModel
        from ..models.base import BaseRecommender

        ensemble = EnsembleModel()
        weights_config = Config.get("models.ensemble.weights", {})
        for model_name in ["popularity", "svd", "ncf", "lightgcn", "content_bert"]:
            try:
                sub_model = BaseRecommender.load(model_name)
                w = weights_config.get(model_name, 0.2)
                ensemble.add_model(sub_model, weight=w)
            except FileNotFoundError:
                self.logger.warning(f"子模型 {model_name} 不可用，跳过")

        ensemble.train(train_df, val_df)
        metrics = ensemble.evaluate(test_df)
        ensemble.save()

        self.model_results["ensemble"] = {
            "model": "ensemble",
            "metrics": metrics,
            "weights": ensemble.weights,
        }
        self.logger.info(f"[Ensemble] 指标: {metrics}")

    def _select_best_model(self) -> Dict:
        """基于NDCG@10选择最佳模型"""
        best_name = None
        best_ndcg = -1
        for name, result in self.model_results.items():
            if "error" in result:
                continue
            ndcg = result.get("metrics", {}).get("ndcg@10", 0)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_name = name
        return {"best_model": best_name, "best_ndcg@10": best_ndcg}

    def report(self) -> str:
        """生成模型对比报告"""
        lines = [
            "=" * 60,
            "  🤖 建模Agent — 模型训练与评估报告",
            "=" * 60,
            "",
            f"{'模型':<16} {'RMSE':<8} {'MAE':<8} {'NDCG@10':<9} {'HR@10':<8} {'训练时间':<10}",
            "-" * 60,
        ]

        for name, r in self.model_results.items():
            if name.startswith("_") or "error" in r:
                continue
            m = r.get("metrics", {})
            lines.append(
                f"{name:<16} "
                f"{m.get('rmse', 0):<8.4f} "
                f"{m.get('mae', 0):<8.4f} "
                f"{m.get('ndcg@10', 0):<9.4f} "
                f"{m.get('hit_rate@10', 0):<8.4f} "
                f"{r.get('training_time', 0):<10.1f}s"
            )

        lines.append("-" * 60)

        if "_best" in self.model_results:
            best = self.model_results["_best"]
            lines.append(f"\n🏆 最佳模型: {best.get('best_model')} (NDCG@10={best.get('best_ndcg@10', 0):.4f})")

        if "ensemble" in self.model_results:
            e = self.model_results["ensemble"]
            m = e.get("metrics", {})
            lines.append(f"\n🎯 Ensemble: NDCG@10={m.get('ndcg@10', 0):.4f} | 权重={e.get('weights', {})}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
