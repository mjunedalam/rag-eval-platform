"""Tests for the API app through FastAPI's TestClient, with fake pipelines."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from tests.unit.test_api_schemas import SOURCE

from rag_eval_platform.api.main import create_app
from rag_eval_platform.api.schemas import CheckOut
from rag_eval_platform.api.security import RateLimiter
from rag_eval_platform.api.services import ApiServices
from rag_eval_platform.generation.generator import Answer, Citation, GenerationError
from rag_eval_platform.pipeline import TimedAnswer
from rag_eval_platform.retrieval.vector_store import VectorStoreError


def answer(text: str = "F equals ma [1].") -> Answer:
    return Answer(question="q?", text=text, citations=(Citation(1, SOURCE),),
                  invalid_citations=(), sources=(SOURCE,), model="fake", prompt_version="v1",
                  input_tokens=10, output_tokens=4, latency_ms=20.0)  # fmt: skip


class FakeStream:
    def __init__(self, pieces: list[str], fail: bool = False) -> None:
        self.pieces, self.fail = pieces, fail
        self.answer = answer("".join(pieces))

    def __iter__(self) -> Iterator[str]:
        yield from self.pieces
        if self.fail:
            raise GenerationError("model crashed")


class FakePipeline:
    def __init__(self, error: Exception | None = None, fail_stream: bool = False) -> None:
        self.error, self.fail_stream = error, fail_stream
        self.asked: list[str] = []

    def ask_timed(self, question: str) -> TimedAnswer:
        self.asked.append(question)
        if self.error:
            raise self.error
        return TimedAnswer(answer(), 5.0, 30.0)

    def ask_stream(self, question: str) -> FakeStream:
        if self.error:
            raise self.error
        return FakeStream(["F equals ", "ma [1]."], fail=self.fail_stream)


def client(pipeline: FakePipeline | None = None, *, key: str | None = None, limit: int = 30,
           healthy: bool = True) -> TestClient:  # fmt: skip
    pipe = pipeline or FakePipeline()
    services = ApiServices(
        pipelines={"concise": pipe, "detailed": pipe},
        checks={"chroma": lambda: CheckOut(ok=True, detail="40 chunks"),
                "llm": lambda: CheckOut(ok=healthy, detail="qwen3:8b" if healthy else "down")},
        api_key=key, limiter=RateLimiter(limit),
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
    api = client(key="s3cret")

    assert api.post("/query", json={"question": "q"}).status_code == 401
    assert (
        api.post("/query", json={"question": "q"}, headers={"X-API-Key": "nope"}).status_code == 401
    )
    assert (
        api.post("/query", json={"question": "q"}, headers={"X-API-Key": "s3cret"}).status_code
        == 200
    )


def test_rate_limit_answers_429_with_retry_after() -> None:
    api = client(limit=1)

    assert api.post("/query", json={"question": "q"}).status_code == 200
    limited = api.post("/query", json={"question": "q"})
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


@pytest.mark.parametrize(("error", "status"), [
    (GenerationError("LLM down. Hint: ollama serve"), 502),
    (VectorStoreError("Chroma down"), 503),
])  # fmt: skip
def test_backend_failures_become_clean_json_errors(error: Exception, status: int) -> None:
    response = client(FakePipeline(error=error)).post("/query", json={"question": "q"})

    assert response.status_code == status
    assert set(response.json()) == {"error", "hint"}
    assert "Traceback" not in response.text


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
    assert "model crashed" in json.loads(got[-1][1])["error"]
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
