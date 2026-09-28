"""Instant answer health: how well an answer is supported, measured in under a millisecond.

The LLM judge (⑤ Evaluate) takes one to two minutes. These signals need no model at all,
so every answer gets them the moment it finishes:

- citation coverage: the share of claims (sentences, bullets, table rows) with a citation;
- grounding: for each cited claim, the share of its content words found in the sources it
  cites (a word-overlap proxy for faithfulness: high means the wording comes from the
  sources, low means the claim may be the model's own);
- sources, context size, answer length, and where the time went.
"""

import html
import re
from dataclasses import dataclass
from typing import Literal

from rag_eval_platform.generation.generator import CITATION_PATTERN
from rag_eval_platform.playground.core import QueryTrace

Band = Literal["good", "fair", "poor", "none"]

MIN_CLAIM_WORDS = 3  # shorter lines ("Ok.", a table cell) are not claims
GOOD, FAIR = 0.8, 0.5
_WORD = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "with",
        "that",
        "this",
        "from",
        "into",
        "its",
        "has",
        "have",
        "had",
        "not",
        "but",
        "can",
        "may",
        "also",
        "than",
        "then",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "each",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "very",
        "will",
        "would",
        "should",
        "could",
        "about",
        "over",
        "under",
        "between",
        "these",
        "those",
        "there",
        "their",
        "they",
        "them",
        "our",
        "your",
        "you",
        "his",
        "her",
        "she",
        "him",
        "it's",
        "is",
        "of",
        "to",
        "in",
        "on",
        "a",
        "an",
        "as",
        "at",
        "by",
        "be",
        "or",
        "if",
        "so",
        "do",
        "does",
        "did",
    }
)


@dataclass(frozen=True)
class AnswerHealth:
    citation_coverage: float | None  # None for a refusal or an answer without claims
    grounding: float | None  # None when no claim cites a source
    distinct_docs: int
    distinct_pages: int
    context_chars: int
    prompt_tokens: int | None
    answer_words: int
    retrieve_s: float
    first_token_s: float | None
    write_s: float | None


@dataclass(frozen=True)
class HealthRow:
    label: str
    value: str
    detail: str
    score: float | None  # 0..1 for a bar, None for a count
    band: Band


def claims(text: str) -> tuple[str, ...]:
    """Sentences, bullets and table rows that state something (headings and rules skipped)."""
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or _TABLE_RULE.match(line):
            continue
        for sentence in _SENTENCE_END.split(line):
            if len(_WORD.findall(CITATION_PATTERN.sub("", sentence))) >= MIN_CLAIM_WORDS:
                found.append(sentence.strip())
    return tuple(found)


def answer_health(trace: QueryTrace, first_token_s: float | None) -> AnswerHealth:
    answer, sources = trace.answer, trace.final_results
    refused = answer.is_refusal
    statements = () if refused else claims(answer.text)
    cited = [(claim, _cited_numbers(claim, len(sources))) for claim in statements]
    overlaps = [
        _overlap(claim, " ".join(sources[n - 1].chunk.text for n in numbers))
        for claim, numbers in cited
        if numbers
    ]
    latency_s = answer.latency_ms / 1000
    write_s = None
    if not refused:
        write_s = latency_s - first_token_s if first_token_s is not None else latency_s
    return AnswerHealth(
        citation_coverage=(sum(1 for _, n in cited if n) / len(cited)) if cited else None,
        grounding=(sum(overlaps) / len(overlaps)) if overlaps else None,
        distinct_docs=len({s.chunk.doc_id for s in sources}),
        distinct_pages=len({(s.chunk.doc_id, s.chunk.page) for s in sources}),
        context_chars=sum(len(s.chunk.text) for s in sources),
        prompt_tokens=answer.input_tokens,
        answer_words=len(_WORD.findall(CITATION_PATTERN.sub("", answer.text))),
        retrieve_s=trace.retrieval_ms / 1000,
        first_token_s=first_token_s,
        write_s=write_s,
    )


def band(score: float | None) -> Band:
    if score is None:
        return "none"
    return "good" if score >= GOOD else "fair" if score >= FAIR else "poor"


def health_rows(health: AnswerHealth) -> tuple[HealthRow, ...]:
    def percent(score: float | None) -> str:
        return "-" if score is None else f"{score:.0%}"

    def seconds(value: float | None) -> str:
        return "-" if value is None else f"{value:.1f} s"

    parts = (health.retrieve_s, health.first_token_s, health.write_s)
    total = sum(p for p in parts if p is not None)
    documents = f"{health.distinct_docs} document{'s' if health.distinct_docs != 1 else ''}"
    pages = f"{health.distinct_pages} page{'s' if health.distinct_pages != 1 else ''}"
    tokens = f" · {health.prompt_tokens:,} prompt tokens" if health.prompt_tokens else ""
    return (
        HealthRow("coverage", percent(health.citation_coverage),
                  "claims that cite a source", health.citation_coverage,
                  band(health.citation_coverage)),
        HealthRow("grounding", percent(health.grounding),
                  "cited wording found in its source", health.grounding,
                  band(health.grounding)),
        HealthRow("sources", pages, f"from {documents}", None, "none"),
        HealthRow("context", f"{health.context_chars:,} chars", f"sent to the model{tokens}",
                  None, "none"),
        HealthRow("answer", f"{health.answer_words:,} words", "without citation markers",
                  None, "none"),
        HealthRow("time", f"{total:.1f} s",
                  f"{seconds(parts[0])} search + {seconds(parts[1])} first token + "
                  f"{seconds(parts[2])} writing", None, "none"),
    )  # fmt: skip


def health_html(rows: tuple[HealthRow, ...]) -> str:
    """Scorecards: a big value, its label, a short explanation, and a bar for scores."""
    cards = []
    for row in rows:
        bar = ""
        if row.score is not None:
            bar = f'<div class="bar"><span style="width:{row.score:.0%}"></span></div>'
        cards.append(
            f'<div class="rag-health-card {row.band}" title="{html.escape(row.detail)}">'
            f'<div class="v">{html.escape(row.value)}'
            f'</div><div class="l">{html.escape(row.label)}</div>{bar}'
            f'<div class="d">{html.escape(row.detail)}</div></div>'
        )
    return f'<div class="rag-health">{"".join(cards)}</div>'


def _cited_numbers(claim: str, sources: int) -> tuple[int, ...]:
    numbers = []
    for match in CITATION_PATTERN.finditer(claim):
        numbers += [int(part) for part in match.group(1).split(",")]
    return tuple(n for n in dict.fromkeys(numbers) if 1 <= n <= sources)


def _overlap(claim: str, source: str) -> float:
    words = _content_words(CITATION_PATTERN.sub("", claim))
    if not words:
        return 1.0
    return len(words & _content_words(source)) / len(words)


def _content_words(text: str) -> set[str]:
    return {w for w in (t.lower() for t in _WORD.findall(text))
            if len(w) >= 3 and w not in _STOPWORDS}  # fmt: skip
