"""The real app from settings, against Chroma (skips without Chroma or the extra)."""

import chromadb
import pytest
from fastapi.testclient import TestClient

from rag_eval_platform.api.main import create_app_from_settings
from rag_eval_platform.config.settings import Settings

pytestmark = pytest.mark.integration


def test_health_reports_each_service() -> None:
    settings = Settings()
    try:
        chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port).heartbeat()
    except Exception:
        pytest.skip("no Chroma server")
    pytest.importorskip("sentence_transformers")

    response = TestClient(create_app_from_settings(settings)).get("/health")

    assert response.status_code in (200, 503)
    assert set(response.json()["checks"]) == {"chroma", "llm"}
    assert response.json()["checks"]["chroma"]["ok"] is True
