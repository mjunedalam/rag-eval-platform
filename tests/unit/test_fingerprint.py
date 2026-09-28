"""Tests for evaluation.fingerprint: what the scores depend on, and what changed."""

from pathlib import Path
from typing import Any

import pytest

from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.evaluation.citation_validity import CITATION_PROMPT_VERSION
from rag_eval_platform.evaluation.fingerprint import (
    MISSING,
    FieldChange,
    current_fingerprint,
    fingerprint_changes,
    tree_sha256,
)
from rag_eval_platform.generation.prompt_templates import PROMPT_VERSION


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "a.md").write_text("alpha", encoding="utf-8")
    golden = tmp_path / "qa.json"
    golden.write_text("[]", encoding="utf-8")
    return get_settings().model_copy(
        update={"raw_data_dir": raw, "golden_dataset_path": golden, **overrides}
    )


def test_fingerprint_records_settings_versions_and_hashes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    fp = current_fingerprint(settings)

    assert fp.chunk_size == settings.chunk_size
    assert fp.embedding_model == settings.embedding_model
    assert fp.prompt_version == PROMPT_VERSION
    assert fp.citation_prompt_version == CITATION_PROMPT_VERSION
    assert len(fp.golden_sha256) == len(fp.corpus_sha256) == 64


def test_corpus_hash_ignores_hidden_files_but_sees_edits_and_renames(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    raw = settings.raw_data_dir
    first = tree_sha256(raw)

    (raw / ".DS_Store").write_bytes(b"finder noise")
    assert tree_sha256(raw) == first

    (raw / "a.md").rename(raw / "b.md")
    renamed = tree_sha256(raw)
    assert renamed != first

    (raw / "b.md").write_text("changed", encoding="utf-8")
    assert tree_sha256(raw) != renamed


def test_corpus_hash_needs_the_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tree_sha256(tmp_path / "nope")


def test_reranker_details_only_count_while_reranking(tmp_path: Path) -> None:
    off_20 = current_fingerprint(make_settings(tmp_path, rerank=False, rerank_candidates=20))
    off_30 = current_fingerprint(make_settings(tmp_path, rerank=False, rerank_candidates=30))
    on_20 = current_fingerprint(make_settings(tmp_path, rerank=True, rerank_candidates=20))
    on_30 = current_fingerprint(make_settings(tmp_path, rerank=True, rerank_candidates=30))

    assert off_20 == off_30
    assert on_20 != on_30


def test_changes_list_each_differing_field(tmp_path: Path) -> None:
    fp = current_fingerprint(make_settings(tmp_path))
    saved = {**fp.to_dict(), "chunk_size": fp.chunk_size + 1}

    changes = fingerprint_changes(saved, fp)

    assert changes == (FieldChange("chunk_size", fp.chunk_size + 1, fp.chunk_size),)
    assert str(changes[0]) == f"chunk_size: {fp.chunk_size + 1} → {fp.chunk_size}"
    assert fingerprint_changes(fp.to_dict(), fp) == ()


def test_missing_or_unknown_fields_count_as_changes(tmp_path: Path) -> None:
    fp = current_fingerprint(make_settings(tmp_path))
    saved = {k: v for k, v in fp.to_dict().items() if k != "top_k"} | {"old_field": 1}

    changes = {c.field: c for c in fingerprint_changes(saved, fp)}

    assert set(changes) == {"top_k", "old_field"}
    assert changes["top_k"].saved == MISSING
    assert changes["old_field"].current == MISSING
