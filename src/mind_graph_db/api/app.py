"""FastAPI application factory for Mind Graph DB REST service."""

from typing import Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mind_graph_db.api.deps import init_database
from mind_graph_db.api.routes import router


def create_app(db_dir: Optional[str] = None) -> FastAPI:
    """Create and configure FastAPI web application instance.

    Args:
        db_dir: Optional custom database storage directory path.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Mind Graph DB API",
        description="Hybrid Semantic + Vector + Graph Database REST Service",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if db_dir:
        init_database(db_dir=db_dir)

    app.include_router(router, prefix="/api/v1")
    app.include_router(router)

    @app.get("/health")
    def health_check() -> Dict[str, str]:
        return {"status": "ok", "service": "Mind Graph DB"}

    return app

