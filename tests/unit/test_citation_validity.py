"""Tests for evaluation.citation_validity with a fake judge LLM."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation import citation_validity as cv_module
from rag_eval_platform.evaluation.citation_validity import (
    CITATION_PROMPT_VERSION,
    CitationChecker,
    CitedClaim,
    build_citation_messages,
    extract_cited_claims,
    parse_verdicts,
)
from rag_eval_platform.evaluation.judge import JudgeError
from rag_eval_platform.generation.generator import (
    Answer,
    Completion,
    Generator,
    OpenAICompatibleClient,
)
from rag_eval_platform.generation.prompt_templates import Message
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult


@dataclass
class FakeLlm:
    replies: list[str]
    model: str = "judge-model"
    received: list[Sequence[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message]) -> Completion:
        self.received.append(messages)
        return Completion(text=self.replies.pop(0), input_tokens=None, output_tokens=None)

    def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
        raise NotImplementedError


def source(doc_id: str, text: str, n: int = 0) -> SearchResult:
    chunk = Chunk(id=f"{doc_id}#{n}", doc_id=doc_id, index=n, text=text, start_index=0)
    return SearchResult(chunk=chunk, score=0.5)


SOURCES = [
    source("mrr.md", "MRR is the mean of 1/rank of the first relevant document."),
    source("recall.md", "Recall@k is the share of relevant documents in the top k."),
    source("ndcg.md", "NDCG rewards relevant documents near the top."),
]


def answer_with(text: str) -> Answer:
    return Generator(FakeLlm([text])).generate("What is MRR?", SOURCES)


class TestExtractCitedClaims:
    def test_keeps_sentences_with_valid_citations(self) -> None:
        text = "MRR averages 1/rank [1]. Recall counts hits [2][3]. No cite here. Bad [9]."

        claims = extract_cited_claims(text, source_count=3)

        assert claims == (
            CitedClaim("MRR averages 1/rank.", (1,)),
            CitedClaim("Recall counts hits.", (2, 3)),
        )

    def test_citation_after_the_full_stop_belongs_to_the_previous_sentence(self) -> None:
        claims = extract_cited_claims("MRR averages 1/rank. [1] Recall counts [2].", 3)

        assert claims == (
            CitedClaim("MRR averages 1/rank.", (1,)),
            CitedClaim("Recall counts.", (2,)),
        )

    def test_grouped_citations_and_repeats(self) -> None:
        claims = extract_cited_claims("Both metrics rank [1, 2] documents [1].", 3)

        assert claims == (CitedClaim("Both metrics rank documents.", (1, 2)),)

    def test_no_citations_gives_no_claims(self) -> None:
        assert extract_cited_claims("Nothing is cited here.", 3) == ()


def test_messages_pair_each_claim_with_its_source_text() -> None:
    system, user = build_citation_messages(
        [("MRR averages 1/rank", SOURCES[0].chunk.text), ("Recall counts hits", "Recall text")]
    )

    assert system.role == "system"
    assert "JSON" in system.content
    assert user.role == "user"
    assert "Item 1" in user.content
    assert "Item 2" in user.content
    assert "CLAIM: MRR averages 1/rank" in user.content
    assert "SOURCE: MRR is the mean" in user.content
    assert "[1]" not in user.content
    assert "exactly 2 verdicts" in user.content


class TestParseVerdicts:
    def test_plain_json(self) -> None:
        assert parse_verdicts('{"verdicts": ["yes", "no"]}', expected=2) == (True, False)

    def test_json_inside_a_code_fence_and_mixed_case(self) -> None:
        text = 'Sure!\n```json\n{"verdicts": ["Yes", " NO "]}\n```'

        assert parse_verdicts(text, expected=2) == (True, False)

    @pytest.mark.parametrize(
        "text",
        [
            "no json at all",
            '{"verdicts": ["yes"]}',
            '{"verdicts": ["yes", "maybe"]}',
            '{"other": ["yes", "no"]}',
            '{"verdicts": "yes, no"}',
        ],
    )
    def test_rejects_malformed_replies(self, text: str) -> None:
        with pytest.raises(JudgeError):
            parse_verdicts(text, expected=2)


class TestCitationChecker:
    def test_scores_share_of_supported_citations(self) -> None:
        judge = FakeLlm(['{"verdicts": ["yes", "yes", "no"]}'])
        answer = answer_with("MRR averages 1/rank [1]. Recall counts hits [2][3].")

        validity = CitationChecker(judge).check(answer)

        assert validity.score == pytest.approx(2 / 3)
        assert [(v.number, v.doc_id, v.supported) for v in validity.verdicts] == [
            (1, "mrr.md", True),
            (2, "recall.md", True),
            (3, "ndcg.md", False),
        ]
        assert validity.unsupported == ((3, "Recall counts hits."),)
        _, user = judge.received[0]
        assert "Item 3" in user.content
        assert "NDCG rewards" in user.content

    def test_answer_without_citations_is_not_scored_and_skips_the_judge(self) -> None:
        judge = FakeLlm([])

        validity = CitationChecker(judge).check(answer_with("Nothing is cited here."))

        assert validity.score is None
        assert judge.received == []

    def test_retries_once_on_an_unreadable_reply(self) -> None:
        judge = FakeLlm(["I think yes", '{"verdicts": ["no"]}'])

        validity = CitationChecker(judge).check(answer_with("MRR averages 1/rank [1]."))

        assert validity.score == 0.0
        assert len(judge.received) == 2

    def test_wrong_count_twice_falls_back_to_one_request_per_citation(self) -> None:
        batch = '```json\n{"verdicts": ["yes"]}\n```'  # 1 verdict for 2 citations
        judge = FakeLlm([batch, batch, '{"verdicts": ["yes"]}', '{"verdicts": ["no"]}'])

        validity = CitationChecker(judge).check(answer_with("Both rank documents [1][2]."))

        assert [v.supported for v in validity.verdicts] == [True, False]
        assert len(judge.received) == 4
        single = judge.received[2][1].content
        assert "Item 1" in single
        assert "Item 2" not in single

    def test_gives_up_when_even_a_single_citation_is_unreadable(self) -> None:
        judge = FakeLlm(["nope", "still nope"])

        with pytest.raises(JudgeError, match="citation"):
            CitationChecker(judge).check(answer_with("MRR averages 1/rank [1]."))


def test_create_citation_checker_connects_to_the_judge_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_connect(settings: Settings, **kwargs: Any) -> FakeLlm:
        captured.update(kwargs)
        return FakeLlm([], model=kwargs["model"])

    monkeypatch.setattr(OpenAICompatibleClient, "connect", staticmethod(fake_connect))

    checker = cv_module.create_citation_checker(Settings(judge_max_tokens=512))

    assert checker.model == "gemma3:12b"
    assert captured["provider"] == "ollama"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 512
    assert captured["timeout"] == 300.0
    assert CITATION_PROMPT_VERSION == "v1"
