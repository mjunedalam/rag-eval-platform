"""Opt-in round trip to a real Langfuse project.

Runs only with Langfuse keys in .env (or the environment) AND ``RAG_TEST_LANGFUSE=1``, so an
ordinary test run never sends anything to Langfuse Cloud:

    RAG_TEST_LANGFUSE=1 uv run pytest tests/integration/test_langfuse_cloud.py -v
"""

import os

import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability import langfuse_tracer
from rag_eval_platform.observability.langfuse_tracer import LangfuseTracer, create_tracer

pytestmark = pytest.mark.integration


@pytest.fixture
def real_tracer(monkeypatch: pytest.MonkeyPatch) -> LangfuseTracer:
    if os.environ.get("RAG_TEST_LANGFUSE") != "1":
        pytest.skip("set RAG_TEST_LANGFUSE=1 to send a test trace to Langfuse")
    pytest.importorskip("langfuse")
    monkeypatch.setenv("RAG_LANGFUSE_ENABLED", "true")  # conftest switches it off for safety
    settings = Settings()
    if not settings.langfuse_ready:
        pytest.skip("no Langfuse keys configured")
    langfuse_tracer.clear_tracer_cache()
    tracer = create_tracer(settings)
    assert isinstance(tracer, LangfuseTracer)
    return tracer


def test_a_trace_and_a_score_reach_langfuse(real_tracer: LangfuseTracer) -> None:
    trace = real_tracer.start("integration-test-trace", "integration test question",
                              tags=["integration-test"], metadata={"purpose": "test"})  # fmt: skip
    trace.stage("retrieve-context", input="integration test question").end(output=[])
    trace.finish(answer="integration test answer")
    real_tracer.score("integration-test-trace", "user_feedback", 1, data_type="BOOLEAN")

    real_tracer.flush()  # raises nothing: data was handed to Langfuse's exporter

    assert trace.llm_ref is not None
    assert real_tracer.trace_url("integration-test-trace")
