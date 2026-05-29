"""Tests for evaluation metrics — pure functions, no data dependencies."""
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_rmse,
    compute_mae,
    dcg_at_k,
    ndcg_at_k,
    evaluate_recommendations,
    build_test_dict,
)


class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_rmse(y, y) == 0.0

    def test_off_by_one(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert compute_rmse(y_true, y_pred) == 1.0


class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_mae(y, y) == 0.0

    def test_off_by_one(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert compute_mae(y_true, y_pred) == 1.0


class TestDCG:
    def test_empty(self):
        assert dcg_at_k([], 10) == 0.0

    def test_all_relevant(self):
        scores = [3.0, 2.0, 1.0]
        result = dcg_at_k(scores, 3)
        assert result > 0

    def test_k_larger_than_list(self):
        scores = [1.0, 0.5]
        result = dcg_at_k(scores, 5)
        assert result > 0


class TestNDCG:
    def test_perfect_ranking(self):
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = ndcg_at_k(scores, 5)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_reversed_ranking(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ndcg_at_k(scores, 5)
        assert 0 < result < 1.0

    def test_empty(self):
        assert ndcg_at_k([], 5) == 0.0


class TestEvaluateRecommendations:
    def test_basic(self):
        predictions = {
            0: [(0, 0.9), (1, 0.8), (2, 0.7)],
            1: [(3, 0.9), (4, 0.8), (0, 0.7)],
        }
        test_data = {0: {0, 2}, 1: {3, 5}}
        metrics = evaluate_recommendations(predictions, test_data, k=3)

        assert "ndcg@3" in metrics
        assert "hit_rate@3" in metrics
        assert "precision@3" in metrics
        assert "recall@3" in metrics
        assert 0 <= metrics["ndcg@3"] <= 1.0
        assert 0 <= metrics["hit_rate@3"] <= 1.0

    def test_all_hits(self):
        predictions = {0: [(0, 0.9), (1, 0.8)]}
        test_data = {0: {0, 1}}
        metrics = evaluate_recommendations(predictions, test_data, k=2)

        assert metrics["hit_rate@2"] == 1.0
        assert metrics["precision@2"] == 1.0
        assert metrics["recall@2"] == 1.0
        assert metrics["ndcg@2"] == 1.0

    def test_no_hits(self):
        predictions = {0: [(5, 0.9), (6, 0.8)]}
        test_data = {0: {0, 1}}
        metrics = evaluate_recommendations(predictions, test_data, k=2)

        assert metrics["hit_rate@2"] == 0.0
        assert metrics["precision@2"] == 0.0
        assert metrics["recall@2"] == 0.0


class TestBuildTestDict:
    def test_basic(self, sample_ratings_df):
        result = build_test_dict(sample_ratings_df, user_col="user_idx",
                                 item_col="movie_idx", rating_col="rating")
        assert 0 in result
        assert 1 in result
        assert 0 in result[0]  # movie 0 rated 5.0 >= 3.5

    def test_threshold_filters_low_ratings(self, sample_ratings_df):
        result = build_test_dict(sample_ratings_df, threshold=4.5)
        # Movie 1 was rated 3.0, should be excluded
        all_movies = set()
        for movies in result.values():
            all_movies.update(movies)
        assert 1 not in all_movies


@pytest.fixture
def sample_ratings_df():
    import pandas as pd
    return pd.DataFrame({
        "user_idx": [0, 0, 1, 1],
        "movie_idx": [0, 1, 0, 2],
        "rating": [5.0, 3.0, 4.0, 4.5],
        "timestamp": [1000, 1001, 1002, 1003],
    })
