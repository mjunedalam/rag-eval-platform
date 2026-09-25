"""Tests for ingestion.cli (the logic behind scripts/run_ingestion.py)."""

import json
from pathlib import Path

import pytest

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.ingestion.chunk_io import load_chunks
from rag_eval_platform.ingestion.cli import IngestionSummary, ingest, main


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text("Alpha paragraph.\n\nSecond alpha paragraph.", encoding="utf-8")
    (raw / "b.txt").write_text("Beta.", encoding="utf-8")
    return raw


def test_ingest_writes_chunks_and_returns_summary(corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "processed" / "chunks.jsonl"

    summary = ingest(corpus, output, strategy="recursive", chunk_size=20, chunk_overlap=0)

    chunks = load_chunks(output)
    assert summary == IngestionSummary(
        documents=2,
        chunks=len(chunks),
        mean_chunk_chars=sum(len(c.text) for c in chunks) / len(chunks),
        output=output,
    )
    assert {c.doc_id for c in chunks} == {"a.md", "b.txt"}


class TestMain:
    @pytest.fixture(autouse=True)
    def isolated_settings(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        get_settings.cache_clear()

    def test_cli_arguments_override_settings(self, corpus: Path, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"

        exit_code = main(
            [
                "--raw-dir", str(corpus),
                "--output", str(output),
                "--strategy", "fixed",
                "--chunk-size", "10",
                "--chunk-overlap", "2",
            ]
        )  # fmt: skip

        assert exit_code == 0
        assert all(len(c.text) <= 10 for c in load_chunks(output))

    def test_defaults_come_from_settings(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAG_RAW_DATA_DIR", str(corpus))
        monkeypatch.setenv("RAG_PROCESSED_DATA_DIR", str(tmp_path / "processed"))

        assert main([]) == 0
        assert (tmp_path / "processed" / "chunks.jsonl").is_file()

    def test_logs_summary_as_json(
        self, corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--raw-dir", str(corpus), "--output", str(tmp_path / "c.jsonl")])

        lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
        summary = next(line for line in lines if line["message"] == "Ingestion complete")
        assert summary["documents"] == 2

    def test_load_failure_returns_exit_code_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["--raw-dir", str(tmp_path / "missing")])

        assert exit_code == 1
        assert "not a directory" in capsys.readouterr().err

    def test_invalid_overlap_returns_exit_code_1(self, corpus: Path) -> None:
        assert main(["--raw-dir", str(corpus), "--chunk-size", "5", "--chunk-overlap", "5"]) == 1
