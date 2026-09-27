"""Tests for generation.generator with a fake LLM client (no model, no network)."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import SecretStr

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation import generator as generator_module
from rag_eval_platform.generation.generator import (
    Completion,
    GenerationError,
    Generator,
    OpenAICompatibleClient,
    parse_citation_numbers,
    strip_thinking,
)
from rag_eval_platform.generation.prompt_templates import NO_ANSWER, PROMPT_VERSION, Message
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult


def result(doc_id: str) -> SearchResult:
    chunk = Chunk(id=f"{doc_id}#0", doc_id=doc_id, index=0, text=f"about {doc_id}", start_index=0)
    return SearchResult(chunk=chunk, score=0.7)


RESULTS = [result("a.md"), result("b.md"), result("c.md")]


@dataclass
class FakeLlm:
    reply: str
    model: str = "fake-model"
    received: list[Sequence[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message]) -> Completion:
        self.received.append(messages)
        return Completion(text=self.reply, input_tokens=120, output_tokens=15)

    def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
        """Replay the reply in 3-character pieces (so tags get split), then usage."""
        self.received.append(messages)
        for start in range(0, len(self.reply), 3):
            yield Completion(
                text=self.reply[start : start + 3], input_tokens=None, output_tokens=None
            )
        yield Completion(text="", input_tokens=120, output_tokens=15)


class TestParseCitationNumbers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Recall matters [2]. MRR too [1].", (2, 1)),
            ("Both [1][3] and again [1].", (1, 3)),
            ("Grouped [1, 3] and spaced [ 2 ].", (1, 3, 2)),
            ("No citations here.", ()),
            ("Years like 2024 or lists [a] are ignored.", ()),
        ],
    )
    def test_finds_numbers_in_first_appearance_order(
        self, text: str, expected: tuple[int, ...]
    ) -> None:
        assert parse_citation_numbers(text) == expected


class TestStripThinking:
    def test_removes_think_blocks(self) -> None:
        assert strip_thinking("<think>\nreasoning...\n</think>\n\nThe answer [1].") == (
            "The answer [1]."
        )

    def test_leaves_normal_text_alone(self) -> None:
        assert strip_thinking("  Plain answer [2].  ") == "Plain answer [2]."


class TestGenerator:
    def test_answer_maps_citations_to_sources(self) -> None:
        llm = FakeLlm("MRR is the mean reciprocal rank [2]. See also [1].")

        answer = Generator(llm).generate("What is MRR?", RESULTS)

        assert answer.text == "MRR is the mean reciprocal rank [2]. See also [1]."
        assert [(c.number, c.source.chunk.doc_id) for c in answer.citations] == [
            (2, "b.md"),
            (1, "a.md"),
        ]
        assert answer.invalid_citations == ()
        assert answer.sources == tuple(RESULTS)

    def test_citation_to_missing_source_is_flagged(self) -> None:
        answer = Generator(FakeLlm("Claim [1]. Invented [7].")).generate("q?", RESULTS)

        assert [c.number for c in answer.citations] == [1]
        assert answer.invalid_citations == (7,)

    def test_records_model_prompt_version_tokens_and_latency(self) -> None:
        answer = Generator(FakeLlm("A [1].", model="qwen3:8b")).generate("q?", RESULTS)

        assert answer.model == "qwen3:8b"
        assert answer.prompt_version == PROMPT_VERSION
        assert (answer.input_tokens, answer.output_tokens) == (120, 15)
        assert answer.latency_ms >= 0

    def test_sends_system_and_user_messages_with_numbered_sources(self) -> None:
        llm = FakeLlm("A [1].")

        Generator(llm).generate("What is MRR?", RESULTS)

        system, user = llm.received[0]
        assert system.role == "system"
        assert "[3] (source: c.md)" in user.content
        assert user.content.endswith("Question: What is MRR?")

    def test_thinking_is_removed_from_the_answer(self) -> None:
        answer = Generator(FakeLlm("<think>hmm</think>Answer [1].")).generate("q?", RESULTS)

        assert answer.text == "Answer [1]."

    def test_refusal_is_detected(self) -> None:
        answer = Generator(FakeLlm(NO_ANSWER)).generate("q?", RESULTS)

        assert answer.is_refusal
        assert answer.citations == ()

    def test_no_sources_returns_refusal_without_calling_the_model(self) -> None:
        llm = FakeLlm("should not be used")

        answer = Generator(llm).generate("q?", [])

        assert answer.text == NO_ANSWER
        assert answer.is_refusal
        assert llm.received == []

    def test_rejects_blank_question(self) -> None:
        with pytest.raises(ValueError, match="question"):
            Generator(FakeLlm("x")).generate("  ", RESULTS)


@dataclass
class FakeChatCompletions:
    content: str | None = "Answer [1]."
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    stream_pieces: tuple[str, ...] = ("Ans", "wer ", "[1].")
    stream_error: Exception | None = None

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return self._stream()
        message = type("Msg", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        usage = type("Usage", (), {"prompt_tokens": 50, "completion_tokens": 9})()
        return type("Response", (), {"choices": [choice], "usage": usage})()

    def _stream(self) -> Iterator[Any]:
        for piece in self.stream_pieces:
            delta = type("Delta", (), {"content": piece})()
            yield type(
                "Chunk", (), {"choices": [type("C", (), {"delta": delta})()], "usage": None}
            )()
        if self.stream_error is not None:
            raise self.stream_error
        usage = type("Usage", (), {"prompt_tokens": 50, "completion_tokens": 9})()
        yield type("Chunk", (), {"choices": [], "usage": usage})()


class FakeOpenAI:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


class TestOpenAICompatibleClient:
    def test_sends_messages_and_settings_and_reads_reply(self) -> None:
        completions = FakeChatCompletions()
        client = OpenAICompatibleClient(
            FakeOpenAI(completions), model="qwen3:8b", temperature=0.0, max_tokens=256
        )

        completion = client.complete([Message("system", "rules"), Message("user", "q")])

        assert completion == Completion(text="Answer [1].", input_tokens=50, output_tokens=9)
        call = completions.calls[0]
        assert call["model"] == "qwen3:8b"
        assert call["messages"] == [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "q"},
        ]
        assert (call["temperature"], call["max_tokens"]) == (0.0, 256)
        assert "reasoning_effort" not in call

    def test_passes_reasoning_effort_when_set(self) -> None:
        completions = FakeChatCompletions()
        client = OpenAICompatibleClient(
            FakeOpenAI(completions), model="qwen3:8b", temperature=0, max_tokens=5,
            reasoning_effort="none",
        )  # fmt: skip

        client.complete([Message("user", "q")])

        assert completions.calls[0]["reasoning_effort"] == "none"

    def test_empty_reply_becomes_empty_text(self) -> None:
        client = OpenAICompatibleClient(
            FakeOpenAI(FakeChatCompletions(content=None)), model="m", temperature=0, max_tokens=5
        )

        assert client.complete([Message("user", "q")]).text == ""

    def test_request_failure_raises_generation_error_with_hint(self) -> None:
        completions = FakeChatCompletions(error=ConnectionError("refused"))
        client = OpenAICompatibleClient(
            FakeOpenAI(completions), model="qwen3:8b", temperature=0, max_tokens=5,
            hint="Is Ollama running?",
        )  # fmt: skip

        with pytest.raises(GenerationError, match="Is Ollama running"):
            client.complete([Message("user", "q")])


class TestFromSettings:
    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        class FakeModule:
            @staticmethod
            def OpenAI(**kwargs: Any) -> FakeOpenAI:
                captured.update(kwargs)
                return FakeOpenAI(FakeChatCompletions())

        monkeypatch.setattr(generator_module, "import_optional", lambda name, extra: FakeModule)
        return captured

    def test_ollama_uses_local_base_url_and_placeholder_key(self, captured: dict[str, Any]) -> None:
        client = OpenAICompatibleClient.from_settings(
            Settings(llm_provider="ollama", llm_model="qwen3:8b", llm_timeout_seconds=30)
        )

        assert client.model == "qwen3:8b"
        assert client.reasoning_effort == "none"
        assert captured["base_url"] == "http://localhost:11434/v1"
        assert captured["api_key"] == "ollama"
        assert captured["timeout"] == 30

    def test_openai_requires_a_key(self, captured: dict[str, Any]) -> None:
        with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
            OpenAICompatibleClient.from_settings(Settings(llm_provider="openai"))

    def test_openai_uses_key_and_default_base_url(self, captured: dict[str, Any]) -> None:
        client = OpenAICompatibleClient.from_settings(
            Settings(
                llm_provider="openai",
                llm_model="gpt-x",
                openai_api_key=SecretStr("k"),
                llm_reasoning_effort="default",
            )
        )

        assert client.reasoning_effort is None
        assert captured["api_key"] == "k"
        assert captured.get("base_url") is None

    def test_anthropic_is_not_implemented_yet(self, captured: dict[str, Any]) -> None:
        with pytest.raises(NotImplementedError, match="anthropic"):
            OpenAICompatibleClient.from_settings(Settings(llm_provider="anthropic"))


def test_create_generator_wraps_client_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLlm("A [1].", model="from-settings")
    monkeypatch.setattr(OpenAICompatibleClient, "from_settings", staticmethod(lambda s: llm))

    answer = generator_module.create_generator(Settings()).generate("q?", RESULTS)

    assert answer.model == "from-settings"


class TestClientStreaming:
    def test_stream_yields_text_pieces_then_usage(self) -> None:
        completions = FakeChatCompletions()
        client = OpenAICompatibleClient(
            FakeOpenAI(completions), model="qwen3:8b", temperature=0, max_tokens=64,
            reasoning_effort="none",
        )  # fmt: skip

        pieces = list(client.stream([Message("user", "q")]))

        assert pieces == [
            Completion("Ans", None, None),
            Completion("wer ", None, None),
            Completion("[1].", None, None),
            Completion("", 50, 9),
        ]
        call = completions.calls[0]
        assert call["stream"] is True
        assert call["stream_options"] == {"include_usage": True}
        assert call["reasoning_effort"] == "none"

    @pytest.mark.parametrize("where", ["start", "middle"])
    def test_stream_failure_raises_generation_error(self, where: str) -> None:
        completions = (
            FakeChatCompletions(error=ConnectionError("refused"))
            if where == "start"
            else FakeChatCompletions(stream_error=ConnectionError("dropped"))
        )
        client = OpenAICompatibleClient(
            FakeOpenAI(completions), model="m", temperature=0, max_tokens=5, hint="Is Ollama up?"
        )

        with pytest.raises(GenerationError, match="Is Ollama up"):
            list(client.stream([Message("user", "q")]))


class TestAnswerStream:
    def test_pieces_join_to_the_final_answer_with_citations(self) -> None:
        stream = Generator(FakeLlm("MRR is the mean reciprocal rank [2].")).stream("q?", RESULTS)

        pieces = list(stream)

        assert len(pieces) > 1
        assert "".join(pieces) == "MRR is the mean reciprocal rank [2]."
        answer = stream.answer
        assert answer.text == "MRR is the mean reciprocal rank [2]."
        assert [c.source.chunk.doc_id for c in answer.citations] == ["b.md"]
        assert (answer.input_tokens, answer.output_tokens) == (120, 15)
        assert answer.latency_ms >= 0

    def test_thinking_is_never_shown_even_when_tags_are_split(self) -> None:
        stream = Generator(FakeLlm("<think>plan it</think>\n\nAnswer [1].")).stream("q?", RESULTS)

        shown = "".join(stream)

        assert shown == "Answer [1]."
        assert "<" not in shown
        assert stream.answer.text == "Answer [1]."

    def test_answer_is_unavailable_until_the_stream_is_consumed(self) -> None:
        stream = Generator(FakeLlm("A [1].")).stream("q?", RESULTS)

        with pytest.raises(RuntimeError, match="not finished"):
            _ = stream.answer

    def test_no_sources_streams_the_refusal_without_calling_the_model(self) -> None:
        llm = FakeLlm("unused")
        stream = Generator(llm).stream("q?", [])

        assert "".join(stream) == NO_ANSWER
        assert stream.answer.is_refusal
        assert llm.received == []

    def test_on_complete_receives_the_answer(self) -> None:
        finished: list[Any] = []
        stream = Generator(FakeLlm("A [1].")).stream("q?", RESULTS, on_complete=finished.append)

        list(stream)

        assert finished == [stream.answer]

    def test_rejects_blank_question(self) -> None:
        with pytest.raises(ValueError, match="question"):
            Generator(FakeLlm("x")).stream(" ", RESULTS)
