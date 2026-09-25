"""Tests for evaluation.golden_dataset."""

import json
from pathlib import Path
from typing import Any

import pytest

from rag_eval_platform.evaluation.golden_dataset import (
    GoldenDatasetError,
    GoldenExample,
    check_doc_references,
    load_golden_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_example(**overrides: Any) -> dict[str, Any]:
    example: dict[str, Any] = {
        "id": "retrieval-001",
        "question": "What does MRR measure?",
        "expected_answer": "How high the first relevant document ranks.",
        "relevant_doc_ids": ["retrieval_metrics.md"],
        "query_type": "short",
    }
    return {**example, **overrides}


def write_dataset(tmp_path: Path, content: Any) -> Path:
    path = tmp_path / "qa_pairs.json"
    path.write_text(json.dumps(content))
    return path


class TestLoadGoldenDataset:
    def test_loads_valid_examples_in_file_order(self, tmp_path: Path) -> None:
        path = write_dataset(
            tmp_path, [make_example(), make_example(id="retrieval-002", query_type="multi_hop")]
        )

        examples = load_golden_dataset(path)

        assert [e.id for e in examples] == ["retrieval-001", "retrieval-002"]
        assert examples[0] == GoldenExample(
            id="retrieval-001",
            question="What does MRR measure?",
            expected_answer="How high the first relevant document ranks.",
            relevant_doc_ids=("retrieval_metrics.md",),
            query_type="short",
        )

    def test_examples_are_immutable(self, tmp_path: Path) -> None:
        (example,) = load_golden_dataset(write_dataset(tmp_path, [make_example()]))

        with pytest.raises(ValueError, match="frozen"):
            example.question = "changed"  # type: ignore[misc]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenDatasetError, match="not found"):
            load_golden_dataset(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "qa_pairs.json"
        path.write_text("[{not json")

        with pytest.raises(GoldenDatasetError, match="valid JSON"):
            load_golden_dataset(path)

    def test_top_level_must_be_a_list(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenDatasetError, match="list"):
            load_golden_dataset(write_dataset(tmp_path, {"examples": []}))

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenDatasetError, match="empty"):
            load_golden_dataset(write_dataset(tmp_path, []))

    def test_duplicate_ids_raise(self, tmp_path: Path) -> None:
        path = write_dataset(tmp_path, [make_example(), make_example()])

        with pytest.raises(GoldenDatasetError, match="retrieval-001"):
            load_golden_dataset(path)

    @pytest.mark.parametrize(
        "bad_field",
        [
            {"question": "   "},
            {"expected_answer": ""},
            {"relevant_doc_ids": []},
            {"relevant_doc_ids": ["a.md", "a.md"]},
            {"query_type": "unknown"},
            {"extra_field": "x"},
        ],
    )
    def test_invalid_example_reports_its_position(
        self, tmp_path: Path, bad_field: dict[str, Any]
    ) -> None:
        path = write_dataset(tmp_path, [make_example(id="ok-1"), make_example(**bad_field)])

        with pytest.raises(GoldenDatasetError, match="index 1"):
            load_golden_dataset(path)


class TestCheckDocReferences:
    def test_passes_when_all_referenced_docs_exist(self) -> None:
        examples = (GoldenExample.model_validate(make_example()),)

        check_doc_references(examples, {"retrieval_metrics.md", "other.md"})

    def test_lists_every_missing_doc(self) -> None:
        examples = (
            GoldenExample.model_validate(make_example(relevant_doc_ids=["gone.md", "ok.md"])),
            GoldenExample.model_validate(make_example(id="x-2", relevant_doc_ids=["lost.md"])),
        )

        with pytest.raises(GoldenDatasetError, match=r"gone\.md.*lost\.md"):
            check_doc_references(examples, {"ok.md"})


def test_repository_golden_dataset_is_valid_and_matches_corpus() -> None:
    examples = load_golden_dataset(REPO_ROOT / "data" / "golden_dataset" / "qa_pairs.json")
    corpus = {p.name for p in (REPO_ROOT / "data" / "raw").glob("*.md")}

    check_doc_references(examples, corpus)
    assert {e.query_type for e in examples} >= {"short", "paraphrase", "multi_hop"}
    assert corpus <= {doc for e in examples for doc in e.relevant_doc_ids}, (
        "every corpus document should be covered by at least one golden question"
    )
