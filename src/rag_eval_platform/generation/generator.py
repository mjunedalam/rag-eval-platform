"""LLM answer generation over retrieved context.

The generator builds the prompt (numbered sources, see prompt_templates.py), asks the LLM,
and maps every ``[n]`` citation in the answer back to the retrieved chunk it points at.
Citations to numbers that were never shown to the model are reported as invalid.

Answers can also be streamed: ``Generator.stream`` returns an ``AnswerStream`` that yields
text as the model writes it (hidden ``<think>`` reasoning is never shown) and builds the same
``Answer`` once it has been fully consumed.

LLMs are reached through the OpenAI-compatible chat API, which covers a local Ollama
server (default, no key) and OpenAI itself. Needs ``uv sync --extra openai``.
"""

import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from rag_eval_platform._optional import import_optional
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.prompt_templates import (
    NO_ANSWER,
    AnswerStyle,
    Message,
    build_messages,
    prompt_version,
)
from rag_eval_platform.retrieval.vector_store import SearchResult

# "[1]", "[1][3]" and "[1, 3]" all count; "[a]" or bare numbers do not.
CITATION_PATTERN = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
# Reasoning models (e.g. qwen3) may wrap their private reasoning in <think> tags.
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = "<think>"

OLLAMA_PLACEHOLDER_KEY = "ollama"  # Ollama ignores the key, but the client requires one


class GenerationError(Exception):
    """The LLM could not be reached or configured."""


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None
    output_tokens: int | None


class LlmClient(Protocol):
    @property
    def model(self) -> str: ...

    def complete(self, messages: Sequence[Message]) -> Completion: ...

    def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
        """Text pieces as they are generated; the last item carries token usage."""
        ...


@dataclass(frozen=True)
class Citation:
    number: int
    source: SearchResult


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    citations: tuple[Citation, ...]
    invalid_citations: tuple[int, ...]
    sources: tuple[SearchResult, ...]
    model: str
    prompt_version: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float

    @property
    def is_refusal(self) -> bool:
        return self.text.strip() == NO_ANSWER


def parse_citation_numbers(text: str) -> tuple[int, ...]:
    """Citation numbers in order of first appearance, without repeats."""
    numbers = (int(part) for group in CITATION_PATTERN.findall(text) for part in group.split(","))
    return tuple(dict.fromkeys(numbers))


def strip_thinking(text: str) -> str:
    return _THINK_PATTERN.sub("", text).strip()


def _visible_so_far(raw: str) -> str:
    """The part of a partial reply that is safe to show: no reasoning, no half-read tag.

    Trailing whitespace is held back too, so what has been shown is always a prefix of
    the final (stripped) answer.
    """
    text = _THINK_PATTERN.sub("", raw)
    open_at = text.find(_THINK_OPEN)
    if open_at != -1:  # reasoning still in progress
        text = text[:open_at]
    for size in range(len(_THINK_OPEN) - 1, 0, -1):  # "<", "<th", ... might be a tag
        if text.endswith(_THINK_OPEN[:size]):
            text = text[:-size]
            break
    return text.strip()


class AnswerStream:
    """Iterate to receive the answer text piece by piece; then read ``.answer``."""

    def __init__(
        self,
        generator: "Generator",
        question: str,
        sources: tuple[SearchResult, ...],
        on_complete: Callable[[Answer], None] | None,
    ) -> None:
        self._generator = generator
        self._question = question
        self._sources = sources
        self._on_complete = on_complete
        self._answer: Answer | None = None

    @property
    def answer(self) -> Answer:
        if self._answer is None:
            raise RuntimeError("the answer stream has not finished yet; iterate over it first")
        return self._answer

    def __iter__(self) -> Iterator[str]:
        if not self._sources:
            yield NO_ANSWER
            self._finish(NO_ANSWER, usage=None, latency_ms=0.0)
            return

        started = time.perf_counter()
        raw, shown, usage = "", "", None
        messages = build_messages(self._question, self._sources, self._generator.style)
        for piece in self._generator.client.stream(messages):
            if piece.input_tokens is not None or piece.output_tokens is not None:
                usage = piece
            raw += piece.text
            visible = _visible_so_far(raw)
            if len(visible) > len(shown) and visible.startswith(shown):
                yield visible[len(shown) :]
                shown = visible

        final = strip_thinking(raw)
        if len(final) > len(shown) and final.startswith(shown):
            yield final[len(shown) :]
        self._finish(final, usage, (time.perf_counter() - started) * 1000)

    def _finish(self, text: str, usage: Completion | None, latency_ms: float) -> None:
        self._answer = self._generator.build_answer(
            self._question, text, self._sources, usage, latency_ms
        )
        if self._on_complete is not None:
            self._on_complete(self._answer)


class Generator:
    def __init__(self, client: LlmClient, style: AnswerStyle = "concise") -> None:
        self._client = client
        self._style: AnswerStyle = style

    @property
    def style(self) -> AnswerStyle:
        return self._style

    @property
    def client(self) -> LlmClient:
        return self._client

    def generate(self, question: str, results: Sequence[SearchResult]) -> Answer:
        """Answer ``question`` from ``results`` only, with citations mapped to sources."""
        if not question.strip():
            raise ValueError("question must not be blank")
        if not results:
            return self.build_answer(question, NO_ANSWER, (), completion=None, latency_ms=0.0)

        started = time.perf_counter()
        completion = self._client.complete(build_messages(question, results, self._style))
        latency_ms = (time.perf_counter() - started) * 1000
        return self.build_answer(
            question, strip_thinking(completion.text), tuple(results), completion, latency_ms
        )

    def stream(
        self,
        question: str,
        results: Sequence[SearchResult],
        on_complete: Callable[[Answer], None] | None = None,
    ) -> AnswerStream:
        """Like ``generate``, but the answer text arrives piece by piece."""
        if not question.strip():
            raise ValueError("question must not be blank")
        return AnswerStream(self, question, tuple(results), on_complete)

    def build_answer(
        self,
        question: str,
        text: str,
        sources: tuple[SearchResult, ...],
        completion: Completion | None,
        latency_ms: float,
    ) -> Answer:
        numbers = parse_citation_numbers(text)
        return Answer(
            question=question,
            text=text,
            citations=tuple(Citation(n, sources[n - 1]) for n in numbers if 1 <= n <= len(sources)),
            invalid_citations=tuple(n for n in numbers if not 1 <= n <= len(sources)),
            sources=sources,
            model=self._client.model,
            prompt_version=prompt_version(self._style),
            input_tokens=completion.input_tokens if completion else None,
            output_tokens=completion.output_tokens if completion else None,
            latency_ms=latency_ms,
        )


class OpenAICompatibleClient:
    """Chat completions against any OpenAI-compatible server (Ollama, OpenAI, ...)."""

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float,
        max_tokens: int,
        hint: str = "",
        reasoning_effort: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._hint = hint
        self._reasoning_effort = reasoning_effort

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_effort(self) -> str | None:
        return self._reasoning_effort

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Connect to the provider selected by ``RAG_LLM_PROVIDER``."""
        return cls.connect(
            settings,
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            reasoning_effort=(
                None
                if settings.llm_reasoning_effort == "default"
                else settings.llm_reasoning_effort
            ),
        )

    @classmethod
    def connect(
        cls,
        settings: Settings,
        *,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        reasoning_effort: str | None = None,
    ) -> Self:
        """A client for any model on ``provider`` (e.g. the evaluation judge)."""
        options, hint = client_options(settings, provider, model, timeout)
        openai = import_optional("openai", extra="openai")
        return cls(
            openai.OpenAI(**options),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            hint=hint,
            reasoning_effort=reasoning_effort,
        )

    def complete(self, messages: Sequence[Message]) -> Completion:
        try:
            response = self._client.chat.completions.create(**self._request(messages))
        except Exception as exc:  # network, auth, unknown model: all surface as one error
            raise self._error(exc) from exc

        usage = getattr(response, "usage", None)
        return Completion(
            text=response.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )

    def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
        """Yield text pieces as the server sends them, then one item with token usage."""
        try:
            chunks = self._client.chat.completions.create(
                **self._request(messages), stream=True, stream_options={"include_usage": True}
            )
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield Completion(chunk.choices[0].delta.content, None, None)
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    yield Completion("", usage.prompt_tokens, usage.completion_tokens)
        except Exception as exc:  # also covers a connection dropped mid-stream
            raise self._error(exc) from exc

    def _request(self, messages: Sequence[Message]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._reasoning_effort is not None:
            request["reasoning_effort"] = self._reasoning_effort
        return request

    def _error(self, exc: Exception) -> GenerationError:
        return GenerationError(f"LLM request to model '{self._model}' failed: {exc}. {self._hint}")


def client_options(
    settings: Settings, provider: str, model: str, timeout: float
) -> tuple[dict[str, Any], str]:
    """Keyword arguments for an ``openai`` client on ``provider``, plus a fix hint for errors.

    Shared by the sync client above and the async client the RAGAS judge needs.
    """
    if provider == "anthropic":
        raise NotImplementedError("anthropic generation is not implemented yet")
    if provider == "ollama":
        hint = (
            f"Is Ollama running (`ollama serve`) and is the model pulled (`ollama pull {model}`)?"
        )
        options = {
            "base_url": settings.ollama_base_url,
            "api_key": OLLAMA_PLACEHOLDER_KEY,
            "timeout": timeout,
        }
        return options, hint
    if settings.openai_api_key is None:
        raise GenerationError("OPENAI_API_KEY is not set (add it to .env)")
    options = {"api_key": settings.openai_api_key.get_secret_value(), "timeout": timeout}
    return options, "Check OPENAI_API_KEY and the model name."


def create_generator(settings: Settings, style: AnswerStyle = "concise") -> Generator:
    """The generator for a style; the detailed style gets its larger token limit."""
    if style == "detailed":
        settings = settings.model_copy(update={"llm_max_tokens": settings.llm_detailed_max_tokens})
    return Generator(OpenAICompatibleClient.from_settings(settings), style)
