"""推荐系统评估指标 — RMSE, MAE, NDCG@K, HR@K, Precision@K, Recall@K"""
import numpy as np
from collections import defaultdict
from typing import Dict, List


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def dcg_at_k(scores: List[float], k: int) -> float:
    """折损累积增益"""
    scores = np.array(scores[:k], dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(scores) + 2))
    return float(np.sum((2 ** scores - 1) / discounts))


def ndcg_at_k(scores: List[float], k: int) -> float:
    """归一化折损累积增益"""
    ideal = sorted(scores, reverse=True)
    dcg = dcg_at_k(scores, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(
    predictions: Dict[int, List[tuple]],  # user_idx → [(item_idx, score), ...]
    test_data: Dict[int, set],            # user_idx → set of relevant items
    k: int = 10,
) -> Dict[str, float]:
    """批量评估推荐质量 — 返回 NDCG@K, HitRate@K, Precision@K, Recall@K"""
    ndcg_scores = []
    hit = 0
    total_users = 0
    precision_scores = []
    recall_scores = []

    for user_idx, relevant in test_data.items():
        if user_idx not in predictions or len(relevant) == 0:
            continue
        total_users += 1

        pred_items = predictions[user_idx][:k]
        pred_set = {item for item, _ in pred_items}

        # Hit Rate
        if pred_set & relevant:
            hit += 1

        # Precision & Recall
        hits_k = len(pred_set & relevant)
        precision_scores.append(hits_k / k)
        recall_scores.append(hits_k / len(relevant) if len(relevant) > 0 else 0.0)

        # NDCG: 用实际评分(或binary relevance)计算
        relevance_scores = [1.0 if item in relevant else 0.0 for item, _ in pred_items]
        ndcg_scores.append(ndcg_at_k(relevance_scores, k))

    return {
        f"ndcg@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f"hit_rate@{k}": hit / total_users if total_users > 0 else 0.0,
        f"precision@{k}": float(np.mean(precision_scores)) if precision_scores else 0.0,
        f"recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
    }


def build_test_dict(df, user_col="user_idx", item_col="movie_idx",
                    rating_col="rating", threshold=3.5) -> Dict[int, set]:
    """将测试DataFrame转为 {user_idx: set(relevant_items)} 格式"""
    test_dict = defaultdict(set)
    relevant = df[df[rating_col] >= threshold]
    for _, row in relevant.iterrows():
        test_dict[row[user_col]].add(row[item_col])
    return dict(test_dict)
