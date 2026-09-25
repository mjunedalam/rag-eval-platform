"""Golden dataset loading and validation.

The golden dataset (``data/golden_dataset/qa_pairs.json``) is a JSON list of examples;
the format is described in docs/evaluation_methodology.md. Loading fails loudly on any
problem, because a silently skipped example would make the evaluation gate lie.
"""

import json
from collections import Counter
from collections.abc import Iterable, Set
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

QueryType = Literal["short", "paraphrase", "multi_hop", "ambiguous"]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GoldenDatasetError(ValueError):
    """The golden dataset is missing, malformed or inconsistent with the corpus."""


class GoldenExample(BaseModel):
    """One question with its reference answer and relevant source documents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: NonEmptyStr
    question: NonEmptyStr
    expected_answer: NonEmptyStr
    relevant_doc_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    query_type: QueryType

    @field_validator("relevant_doc_ids")
    @classmethod
    def _no_duplicate_doc_ids(cls, doc_ids: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("relevant_doc_ids contains duplicates")
        return doc_ids


def load_golden_dataset(path: Path) -> tuple[GoldenExample, ...]:
    """Load and validate every example, in file order."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenDatasetError(f"Golden dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GoldenDatasetError(f"Golden dataset is not valid JSON: {path}: {exc}") from exc

    if not isinstance(raw, list):
        raise GoldenDatasetError(f"Golden dataset must be a JSON list of examples: {path}")
    if not raw:
        raise GoldenDatasetError(f"Golden dataset is empty: {path}")

    examples = tuple(_parse_example(item, index) for index, item in enumerate(raw))

    duplicates = sorted(id_ for id_, count in Counter(e.id for e in examples).items() if count > 1)
    if duplicates:
        raise GoldenDatasetError(f"Duplicate example ids: {', '.join(duplicates)}")
    return examples


def _parse_example(item: object, index: int) -> GoldenExample:
    try:
        return GoldenExample.model_validate(item)
    except ValidationError as exc:
        raise GoldenDatasetError(f"Invalid golden example at index {index}: {exc}") from exc


def check_doc_references(examples: Iterable[GoldenExample], available_doc_ids: Set[str]) -> None:
    """Raise if any example points at a document that is not in the corpus."""
    missing = sorted(
        {doc for e in examples for doc in e.relevant_doc_ids if doc not in available_doc_ids}
    )
    if missing:
        raise GoldenDatasetError(
            f"Golden dataset references documents not in the corpus: {', '.join(missing)}"
        )
