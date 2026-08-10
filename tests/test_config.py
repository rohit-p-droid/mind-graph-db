"""Tests for configuration settings."""

import pytest
from mind_graph_db.config.settings import Settings, get_settings



def test_default_settings() -> None:
    settings = Settings()
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.embedding_dimension == 384
    assert settings.default_top_k == 10


def test_get_settings_singleton() -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setenv("MIND_GRAPH_DB_ENVIRONMENT", "production")
    monkeypatch.setenv("MIND_GRAPH_DB_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("MIND_GRAPH_DB_EMBEDDING_DIMENSION", "768")

    settings = Settings()
    assert settings.environment == "production"
    assert settings.log_level == "WARNING"
    assert settings.embedding_dimension == 768
