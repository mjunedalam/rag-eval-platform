"""Tests for the API app through FastAPI's TestClient, with fake pipelines."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.fake_tracer import FakeTracer
from tests.unit.observe_records import make_query, make_run
from tests.unit.test_api_schemas import SOURCE

from rag_eval_platform.api import main as api_main
from rag_eval_platform.api.main import create_app
from rag_eval_platform.api.schemas import CheckOut
from rag_eval_platform.api.security import RateLimiter
from rag_eval_platform.api.services import ApiServices
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Answer, Citation, GenerationError
from rag_eval_platform.observability.drift import DriftThresholds
from rag_eval_platform.observability.store import ObserveStore
from rag_eval_platform.pipeline import RagPipeline, TimedAnswer
from rag_eval_platform.retrieval.vector_store import VectorStoreError


def answer(text: str = "F equals ma [1].") -> Answer:
    return Answer(question="q?", text=text, citations=(Citation(1, SOURCE),),
                  invalid_citations=(), sources=(SOURCE,), model="fake", prompt_version="v1",
                  input_tokens=10, output_tokens=4, latency_ms=20.0)  # fmt: skip


class FakeStream:
    def __init__(self, pieces: list[str], fail: bool = False) -> None:
        self.pieces, self.fail = pieces, fail
        self.answer = answer("".join(pieces))
        self.query_id: str | None = None

    def __iter__(self) -> Iterator[str]:
        yield from self.pieces
        if self.fail:
            raise GenerationError("model crashed")
        self.query_id = "stream-id"


class FakePipeline:
    def __init__(self, error: Exception | None = None, fail_stream: bool = False) -> None:
        self.error, self.fail_stream = error, fail_stream
        self.asked: list[str] = []

    def ask_timed(self, question: str) -> TimedAnswer:
        self.asked.append(question)
        if self.error:
            raise self.error
        return TimedAnswer(answer(), 5.0, 30.0, "query-id")

    def ask_stream(self, question: str) -> FakeStream:
        if self.error:
            raise self.error
        return FakeStream(["F equals ", "ma [1]."], fail=self.fail_stream)


def client(pipeline: FakePipeline | None = None, *, key: str | None = None, limit: int = 30,
           healthy: bool = True, store: ObserveStore | None = None,
           drift: DriftThresholds | None = None,
           tracer: FakeTracer | None = None) -> TestClient:  # fmt: skip
    pipe = pipeline or FakePipeline()
    services = ApiServices(
        pipelines={"concise": pipe, "detailed": pipe},
        checks={"chroma": lambda: CheckOut(ok=True, detail="40 chunks"),
                "llm": lambda: CheckOut(ok=healthy, detail="qwen3:8b" if healthy else "down")},
        api_key=key, limiter=RateLimiter(limit), store=store, drift=drift or DriftThresholds(),
        **({"tracer": tracer} if tracer is not None else {}),
    )  # fmt: skip
    return TestClient(create_app(services))


def test_health_is_ok_or_503_and_needs_no_key() -> None:
    ok = client(key="k").get("/health")
    degraded = client(healthy=False).get("/health")

    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"
    assert degraded.status_code == 503
    assert degraded.json()["checks"]["llm"] == {"ok": False, "detail": "down"}


def test_query_returns_the_cited_answer() -> None:
    pipeline = FakePipeline()

    response = client(pipeline).post("/query", json={"question": "  What is F?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "F equals ma [1]."
    assert body["citations"][0]["doc_id"] == "book.pdf"
    assert body["timings"] == {"retrieve_ms": 5.0, "generate_ms": 20.0, "total_ms": 30.0}
    assert pipeline.asked == ["What is F?"]


@pytest.mark.parametrize("body", [{"question": "   "}, {"question": "q", "extra": 1}, {}])
def test_invalid_requests_are_422_without_calling_the_model(body: dict[str, object]) -> None:
    pipeline = FakePipeline()

    response = client(pipeline).post("/query", json=body)

    assert response.status_code == 422
    assert pipeline.asked == []


def test_api_key_is_required_when_configured() -> None:
    api = client(key="fake-key")

    assert api.post("/query", json={"question": "q"}).status_code == 401
    assert (
        api.post("/query", json={"question": "q"}, headers={"X-API-Key": "nope"}).status_code == 401
    )
    assert (
        api.post("/query", json={"question": "q"}, headers={"X-API-Key": "fake-key"}).status_code
        == 200
    )


def test_rate_limit_answers_429_with_retry_after() -> None:
    api = client(limit=1)

    assert api.post("/query", json={"question": "q"}).status_code == 200
    limited = api.post("/query", json={"question": "q"})
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


@pytest.mark.parametrize(("error", "status", "public"), [
    (GenerationError("request to http://10.0.0.5:11434/v1 failed"), 502,
     "The language model is unavailable"),
    (VectorStoreError("Cannot reach Chroma at chroma-internal:8000"), 503,
     "The vector store is unavailable"),
])  # fmt: skip
def test_backend_failures_become_clean_json_errors(
    error: Exception, status: int, public: str
) -> None:
    response = client(FakePipeline(error=error)).post("/query", json={"question": "q"})

    assert response.status_code == status
    assert set(response.json()) == {"error", "hint"}
    assert response.json()["error"] == public
    assert "Traceback" not in response.text


@pytest.mark.parametrize("internal", [
    GenerationError("request to http://10.0.0.5:11434/v1 failed"),
    VectorStoreError("Cannot reach Chroma at chroma-internal:8000"),
])  # fmt: skip
def test_backend_failures_never_reveal_internal_details(
    internal: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        response = client(FakePipeline(error=internal)).post("/query", json={"question": "q"})

    assert str(internal) not in response.text
    assert "10.0.0.5" not in response.text
    assert "chroma-internal" not in response.text
    # the full detail is still available to operators, in the server log
    assert any(str(internal) in str(getattr(r, "error", "")) for r in caplog.records)


def test_the_shared_failure_handler_logs_request_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        client(FakePipeline(error=VectorStoreError("Chroma down"))).post(
            "/query", json={"question": "q"}
        )

    assert any(record.message == "Request failed" for record in caplog.records)
    assert not any(record.message == "Query failed" for record in caplog.records)


def events(text: str) -> list[tuple[str, str]]:
    found = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        name = lines[0].removeprefix("event: ")
        data = "\n".join(line.removeprefix("data: ") for line in lines[1:])
        found.append((name, data))
    return found


def test_stream_sends_status_tokens_then_done() -> None:
    response = client().post("/query/stream", json={"question": "q"})

    assert response.headers["content-type"].startswith("text/event-stream")
    got = events(response.text)
    assert [name for name, _ in got] == ["status", "status", "token", "token", "done"]
    assert got[2][1] == "F equals "
    assert json.loads(got[-1][1])["answer"] == "F equals ma [1]."


def test_a_stream_that_fails_midway_ends_with_an_error_event() -> None:
    response = client(FakePipeline(fail_stream=True)).post("/query/stream", json={"question": "q"})

    got = events(response.text)
    assert got[-1][0] == "error"
    body = json.loads(got[-1][1])
    assert body["error"] == "The language model is unavailable"
    assert body["hint"]
    assert "model crashed" not in response.text  # internal detail stays in the server log
    assert "Traceback" not in response.text


def test_unknown_routes_use_the_error_body() -> None:
    response = client().get("/nope")

    assert response.status_code == 404
    assert set(response.json()) == {"error", "hint"}


def test_an_unexpected_crash_is_a_clean_500() -> None:
    api = TestClient(client(FakePipeline(error=RuntimeError("boom"))).app,
                     raise_server_exceptions=False)  # fmt: skip

    response = api.post("/query", json={"question": "q"})

    assert response.status_code == 500
    assert response.json() == {"error": "Internal error", "hint": "See the server log"}
    assert "boom" not in response.text  # internals stay in the log


@pytest.fixture
def store(tmp_path: Path) -> ObserveStore:
    return ObserveStore(tmp_path / "rag.db")


def test_query_and_stream_return_the_log_id() -> None:
    api = client()

    body = api.post("/query", json={"question": "q"}).json()
    events = api.post("/query/stream", json={"question": "q"}).text

    assert body["query_id"] == "query-id"
    done = events.split("event: done\ndata: ")[1].split("\n")[0]
    assert json.loads(done)["query_id"] == "stream-id"


def test_feedback_is_stored_for_a_logged_query(store: ObserveStore) -> None:
    store.add_query(make_query(1))

    response = client(store=store).post(
        "/feedback", json={"query_id": "q1", "rating": "down", "comment": "  wrong source  "}
    )

    assert response.status_code == 201
    [saved] = store.feedback()
    assert (saved.id, saved.rating, saved.comment, saved.source) == (
        response.json()["id"], 0, "wrong source", "api")  # fmt: skip


def test_feedback_for_an_unknown_query_is_404(store: ObserveStore) -> None:
    response = client(store=store).post("/feedback", json={"query_id": "ghost", "rating": "up"})

    assert response.status_code == 404
    assert response.json() == {
        "error": "Unknown query_id", "hint": "Use the query_id from a /query response"
    }  # fmt: skip


def test_feedback_is_503_when_observability_is_off() -> None:
    response = client(store=None).post("/feedback", json={"query_id": "q1", "rating": "up"})

    assert response.status_code == 503
    assert "RAG_OBSERVE_ENABLED" in response.json()["hint"]


@pytest.mark.parametrize("body", [{"query_id": "q1", "rating": "meh"},
                                  {"query_id": "q1", "rating": "up", "comment": "x" * 501},
                                  {"query_id": "", "rating": "up"},
                                  {"query_id": "q1", "rating": "up", "extra": 1}])  # fmt: skip
def test_invalid_feedback_is_422(store: ObserveStore, body: dict[str, object]) -> None:
    response = client(store=store).post("/feedback", json=body)

    assert response.status_code == 422
    assert response.json()["hint"] == (
        'Send {"query_id": "...", "rating": "up" or "down", "comment": "..."}'
    )


def test_feedback_422_is_not_listed_as_a_502_in_the_openapi_schema() -> None:
    schema = client().get("/openapi.json").json()

    assert "502" not in schema["paths"]["/feedback"]["post"]["responses"]
    assert set(schema["paths"]["/feedback"]["post"]["responses"]) >= {
        "401", "404", "422", "429", "503"
    }  # fmt: skip


def test_feedback_needs_the_key_and_is_rate_limited(store: ObserveStore) -> None:
    store.add_query(make_query(1))
    api = client(store=store, key="fake-key", limit=1)

    assert api.post("/feedback", json={"query_id": "q1", "rating": "up"}).status_code == 401
    assert api.post("/feedback", json={"query_id": "q1", "rating": "up"}).status_code == 429


def test_health_lists_drift_warnings_without_changing_the_status(store: ObserveStore) -> None:
    store.add_eval_run(make_run("gate", False, {"recall": 0.7}))

    body = client(store=store).get("/health")

    assert body.status_code == 200
    assert body.json()["warnings"] == ["The latest gate run failed (2026-09-29 12:00 UTC)."]


def test_health_warns_from_a_bounded_recent_slice_of_a_larger_log(store: ObserveStore) -> None:
    # window=2 needs only the last 4 answers; 6 older rows show the read stays bounded (the
    # "have" count for /health's warnings comes from the same drift_report call, so a wrong
    # total would still compute the right warning here -- this exercises recent_queries /
    # feedback_for end to end through the API, not the "have" message itself).
    for n in range(6):
        store.add_query(make_query(n, total_ms=1000.0))
    store.add_query(make_query(6, total_ms=1000.0))
    store.add_query(make_query(7, total_ms=1000.0))
    store.add_query(make_query(8, total_ms=5000.0))
    store.add_query(make_query(9, total_ms=5000.0))

    body = client(store=store, drift=DriftThresholds(window=2)).get("/health")

    assert body.status_code == 200
    assert body.json()["warnings"] == [
        "p95 latency rose from 1.0 s to 5.0 s (last 2 vs previous 2 answers)."
    ]


def test_a_broken_log_never_breaks_health(tmp_path: Path) -> None:
    (tmp_path / "rag.db").write_bytes(b"not a database" * 100)

    body = client(store=ObserveStore(tmp_path / "rag.db")).get("/health")

    assert body.status_code == 200
    assert body.json()["warnings"] == []


def test_feedback_also_scores_the_trace(store: ObserveStore) -> None:
    store.add_query(make_query(1))
    tracer = FakeTracer()

    response = client(store=store, tracer=tracer).post(
        "/feedback", json={"query_id": "q1", "rating": "down", "comment": "wrong page"}
    )

    assert response.status_code == 201
    assert tracer.scores == [{"query_id": "q1", "name": "user_feedback", "value": 0,
                              "comment": "wrong page", "data_type": "BOOLEAN"}]  # fmt: skip


def test_pending_traces_are_flushed_when_the_server_stops() -> None:
    tracer = FakeTracer()

    with client(tracer=tracer):
        assert tracer.flushed == 0

    assert tracer.flushed == 1


def test_app_from_settings_traces_each_answer_style(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = FakeTracer()
    monkeypatch.setattr(api_main, "create_retriever", lambda settings: object())
    monkeypatch.setattr(api_main, "create_generator", lambda settings, style: object())
    monkeypatch.setattr(api_main, "create_tracer", lambda settings: tracer)
    monkeypatch.setattr(api_main, "create_vector_store", lambda settings: object())

    app = api_main.create_app_from_settings(Settings(observe_enabled=False))

    services: ApiServices = app.state.services
    assert services.tracer is tracer
    for style, pipeline in services.pipelines.items():
        assert isinstance(pipeline, RagPipeline)
        assert pipeline.tracer is tracer
        assert pipeline.tags == ("api", style)
