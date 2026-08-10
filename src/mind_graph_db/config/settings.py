"""
Configuration settings for Mind Graph DB.
"""

import os
from functools import lru_cache
from typing import Any, Dict, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """
        Application configuration settings backed by environment variables.
        """

        environment: str = "development"
        log_level: str = "INFO"
        storage_path: str = "./storage"
        embedding_dimension: int = 384
        default_top_k: int = 10
        embedding_model_name: str = "all-MiniLM-L6-v2"
        nlp_model_name: str = "all-MiniLM-L6-v2"
        max_graph_depth: int = 3

        model_config = SettingsConfigDict(
            env_prefix="MIND_GRAPH_DB_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

except ImportError:
    from pydantic import BaseModel, Field

    class Settings(BaseModel):  # type: ignore[no-redef]
        """Fallback application settings using Pydantic BaseModel."""

        environment: str = Field(default_factory=lambda: os.getenv("MIND_GRAPH_DB_ENVIRONMENT", "development"))
        log_level: str = Field(default_factory=lambda: os.getenv("MIND_GRAPH_DB_LOG_LEVEL", "INFO"))
        storage_path: str = Field(default_factory=lambda: os.getenv("MIND_GRAPH_DB_STORAGE_PATH", "./storage"))
        embedding_dimension: int = Field(
            default_factory=lambda: int(os.getenv("MIND_GRAPH_DB_EMBEDDING_DIMENSION", "384"))
        )
        default_top_k: int = Field(default_factory=lambda: int(os.getenv("MIND_GRAPH_DB_DEFAULT_TOP_K", "10")))
        embedding_model_name: str = Field(
            default_factory=lambda: os.getenv("MIND_GRAPH_DB_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        )
        nlp_model_name: str = Field(
            default_factory=lambda: os.getenv("MIND_GRAPH_DB_NLP_MODEL_NAME", "all-MiniLM-L6-v2")
        )

        max_graph_depth: int = Field(default_factory=lambda: int(os.getenv("MIND_GRAPH_DB_MAX_GRAPH_DEPTH", "3")))


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
