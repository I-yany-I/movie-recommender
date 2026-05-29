"""Tests for Config singleton and YAML loading."""
import os
import sys
from pathlib import Path

import pytest

# Ensure project root in path for isolated test runs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config


class TestConfig:
    def test_singleton(self):
        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_config_loaded_after_import(self):
        """Config auto-loads on import — verify keys exist."""
        assert Config.get("project.name") == "MovieRecommender-Agent"
        assert Config.get("project.version") == "1.0.0"

    def test_nested_get(self):
        assert Config.get("models.svd.n_factors") == 100
        assert Config.get("models.ncf.embedding_dim") == 64
        assert Config.get("models.lightgcn.n_layers") == 3

    def test_default_value(self):
        assert Config.get("nonexistent.key") is None
        assert Config.get("nonexistent.key", "fallback") == "fallback"

    def test_data_config(self):
        assert Config.get("data.movielens.name") == "ml-25m"
        assert "title.basics.tsv.gz" in Config.get("data.imdb.files")

    def test_llm_config(self):
        assert Config.get("llm.provider") == "deepseek"
        assert Config.get("llm.model") == "deepseek-chat"
        assert Config.get("llm.temperature") == 0.7

    def test_evaluation_metrics(self):
        metrics = Config.get("evaluation.metrics")
        assert "rmse" in metrics
        assert "ndcg" in metrics
        assert Config.get("evaluation.top_k") == 10

    def test_agent_configs(self):
        assert Config.get("agents.data_agent.name") == "DataAgent"
        assert Config.get("agents.modeling_agent.name") == "ModelingAgent"
        assert Config.get("agents.conversation_agent.name") == "ConversationAgent"

    def test_ensemble_weights_sum_to_one(self):
        weights = Config.get("models.ensemble.weights")
        total = sum(weights.values())
        assert total == pytest.approx(1.0, rel=1e-2)

    def test_project_root(self):
        root = Config.get_project_root()
        assert root.exists()
        assert (root / "config.yaml").exists()
