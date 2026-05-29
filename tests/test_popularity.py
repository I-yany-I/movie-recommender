"""Tests for PopularityModel — no external data dependencies needed."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.popularity import PopularityModel


class TestPopularityModel:
    @pytest.fixture
    def train_df(self):
        """Synthetic ratings for testing."""
        return pd.DataFrame({
            "user_idx": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "movie_idx": [0, 1, 0, 2, 1, 2, 0, 1, 2, 3],
            "rating": [5.0, 3.0, 4.0, 5.0, 2.0, 4.0, 5.0, 4.0, 3.0, 5.0],
            "timestamp": range(1000, 1010),
        })

    def test_initialization(self):
        model = PopularityModel()
        assert model.name == "popularity"
        assert model.popularity_scores is None

    def test_train_creates_scores(self, train_df):
        model = PopularityModel()
        model.train(train_df)
        assert model.popularity_scores is not None
        assert len(model.popularity_scores) == model.n_items
        assert model.n_users == 5
        assert model.n_items == 4

    def test_predict_returns_array(self, train_df):
        model = PopularityModel()
        model.train(train_df)

        user_ids = np.array([0, 1])
        item_ids = np.array([0, 2])
        preds = model.predict(user_ids, item_ids)
        assert len(preds) == 2
        assert preds.dtype == np.float64

    def test_recommend_returns_top_k(self, train_df):
        model = PopularityModel()
        model.train(train_df)

        recs = model.recommend(user_idx=0, top_k=5)
        assert len(recs) <= 5
        for item_idx, score in recs:
            assert isinstance(item_idx, int)
            assert isinstance(score, float)

    def test_recommend_excludes_seen(self, train_df):
        model = PopularityModel()
        model.train(train_df)

        recs = model.recommend(user_idx=0, top_k=2, exclude_seen=True)
        rec_items = {item for item, _ in recs}
        # User 0 has seen movies 0 and 1 — unseen movies 2, 3 should appear
        assert 0 not in rec_items
        assert 1 not in rec_items
        assert rec_items == {2, 3}

    def test_recommend_scores_descending(self, train_df):
        model = PopularityModel()
        model.train(train_df)

        recs = model.recommend(user_idx=0, top_k=5)
        scores = [s for _, s in recs]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_predict_range(self, train_df):
        model = PopularityModel()
        model.train(train_df)

        preds = model.predict(np.array([0]), np.array([0]))
        assert 0.0 <= preds[0] <= 5.0
