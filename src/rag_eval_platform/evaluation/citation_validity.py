"""Citation validity: does each ``[n]`` point to a source that supports its sentence?

Phase 3 already flags citations to sources that do not exist (``invalid_citations``). This
module checks the ones that do exist: every sentence carrying ``[n]`` is paired with the
text of source ``n``, and the judge model answers yes or no for each pair. All pairs of an
answer go to the judge in one request; if it keeps returning the wrong number of verdicts
(local models sometimes merge items), each pair is asked on its own. The score is the share
of pairs the judge says are supported.

Bump ``CITATION_PROMPT_VERSION`` whenever the prompt wording changes: scores depend on it.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.judge import JudgeError
from rag_eval_platform.generation.generator import (
    Answer,
    LlmClient,
    OpenAICompatibleClient,
    parse_citation_numbers,
)
from rag_eval_platform.generation.prompt_templates import Message

CITATION_PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a strict fact checker.
Each numbered item has a CLAIM and a SOURCE. Answer "yes" if the SOURCE states or directly \
implies the CLAIM, otherwise "no". Judge every item on its own.
Reply with JSON only, one verdict per item, in order: {"verdicts": ["yes", "no", ...]}"""

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_LEADING_CITATIONS = re.compile(r"^(?:\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])+")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class CitedClaim:
    sentence: str  # citation markers removed
    numbers: tuple[int, ...]  # valid source numbers, first-appearance order


@dataclass(frozen=True)
class CitationVerdict:
    sentence: str
    number: int
    doc_id: str
    supported: bool


@dataclass(frozen=True)
class CitationValidity:
    verdicts: tuple[CitationVerdict, ...]

    @property
    def score(self) -> float | None:
        """Share of supported citations; ``None`` when the answer cites nothing."""
        if not self.verdicts:
            return None
        return sum(v.supported for v in self.verdicts) / len(self.verdicts)

    @property
    def unsupported(self) -> tuple[tuple[int, str], ...]:
        return tuple((v.number, v.sentence) for v in self.verdicts if not v.supported)


def extract_cited_claims(text: str, source_count: int) -> tuple[CitedClaim, ...]:
    """Sentences that cite at least one of sources ``1..source_count``."""
    sentences: list[list[str]] = []  # [raw sentence text, ...citation groups]
    for segment in _SENTENCE_END.split(text.strip()):
        leading = _LEADING_CITATIONS.match(segment)
        if leading and sentences:  # "claim. [1] Next claim" -> [1] belongs to "claim."
            sentences[-1].append(leading.group(0))
            segment = segment[leading.end() :]
        if segment.strip():
            sentences.append([segment])

    claims = []
    for raw, *extra in sentences:
        numbers = parse_citation_numbers(" ".join([raw, *extra]))
        valid = tuple(n for n in numbers if 1 <= n <= source_count)
        if valid:
            claims.append(CitedClaim(_clean(raw), valid))
    return tuple(claims)


def build_citation_messages(pairs: Sequence[tuple[str, str]]) -> tuple[Message, Message]:
    """System rules plus one numbered (claim, source text) item per citation."""
    items = [
        f"Item {i}\nCLAIM: {claim}\nSOURCE: {source}"
        for i, (claim, source) in enumerate(pairs, start=1)
    ]
    count = f"There are {len(pairs)} items. Reply with exactly {len(pairs)} verdicts."
    return (
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content="\n\n".join([*items, count])),
    )


def parse_verdicts(text: str, expected: int) -> tuple[bool, ...]:
    """Read ``{"verdicts": ["yes", "no", ...]}`` from the judge's reply (code fences allowed)."""
    match = _JSON_OBJECT.search(text)
    try:
        verdicts = json.loads(match.group(0))["verdicts"] if match else None
    except (json.JSONDecodeError, KeyError, TypeError):
        verdicts = None
    if not isinstance(verdicts, list) or len(verdicts) != expected:
        raise JudgeError(f"expected {expected} citation verdicts, got: {text[:200]!r}")

    words = [str(v).strip().lower() for v in verdicts]
    if any(word not in ("yes", "no") for word in words):
        raise JudgeError(f"citation verdicts must be yes or no, got: {verdicts!r}")
    return tuple(word == "yes" for word in words)


class CitationChecker:
    def __init__(self, client: LlmClient, retries: int = 1) -> None:
        self._client = client
        self._retries = retries

    @property
    def model(self) -> str:
        return self._client.model

    def check(self, answer: Answer) -> CitationValidity:
        claims = extract_cited_claims(answer.text, len(answer.sources))
        cited = [(claim, n, answer.sources[n - 1]) for claim in claims for n in claim.numbers]
        if not cited:
            return CitationValidity(())

        pairs = [(claim.sentence, source.chunk.text) for claim, _, source in cited]
        try:
            supported = self._ask(build_citation_messages(pairs), expected=len(pairs))
        except JudgeError:
            if len(pairs) == 1:
                raise
            # One pair per request: slower, but a single verdict cannot be merged or dropped.
            supported = tuple(
                self._ask(build_citation_messages([pair]), expected=1)[0] for pair in pairs
            )
        return CitationValidity(
            tuple(
                CitationVerdict(claim.sentence, n, source.chunk.doc_id, ok)
                for (claim, n, source), ok in zip(cited, supported, strict=True)
            )
        )

    def _ask(self, messages: tuple[Message, Message], expected: int) -> tuple[bool, ...]:
        # Local judges occasionally wrap or truncate the JSON; one retry usually fixes it.
        for attempt in range(self._retries + 1):
            reply = self._client.complete(messages).text
            try:
                return parse_verdicts(reply, expected)
            except JudgeError as exc:
                if attempt == self._retries:
                    raise JudgeError(
                        f"Unreadable citation verdicts from '{self.model}': {exc}"
                    ) from exc
        raise AssertionError("unreachable")


def create_citation_checker(settings: Settings) -> CitationChecker:
    """A checker on the judge model (``RAG_JUDGE_*``), not the model being evaluated."""
    client = OpenAICompatibleClient.connect(
        settings,
        provider=settings.judge_provider,
        model=settings.judge_model,
        temperature=settings.judge_temperature,
        max_tokens=settings.judge_max_tokens,
        timeout=settings.judge_timeout_seconds,
    )
    return CitationChecker(client)


def _clean(sentence: str) -> str:
    text = " ".join(_CITATION.sub(" ", sentence).split())
    return re.sub(r"\s+([.!?,;:])", r"\1", text)
