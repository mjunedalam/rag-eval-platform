"""What the evaluation scores depend on, so a saved baseline can be checked for freshness.

A generation baseline is judged locally (CI has no GPU) and committed. The gate compares
its fingerprint with the current one: any difference means the scores may no longer hold,
and the baseline must be re-made.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.citation_validity import CITATION_PROMPT_VERSION
from rag_eval_platform.generation.prompt_templates import PROMPT_VERSION

MISSING = "(missing)"


@dataclass(frozen=True)
class EvalFingerprint:
    generator_model: str
    llm_temperature: float
    llm_reasoning_effort: str
    prompt_version: str
    judge_model: str
    citation_prompt_version: str
    embedding_model: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank: bool
    rerank_candidates: int  # 0 when not re-ranking (then it cannot affect the scores)
    reranker_model: str  # "" when not re-ranking
    golden_sha256: str
    corpus_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldChange:
    field: str
    saved: Any
    current: Any

    def __str__(self) -> str:
        return f"{self.field}: {self.saved} → {self.current}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    """One hash over every file's relative path and bytes; hidden files are ignored."""
    if not root.is_dir():
        raise FileNotFoundError(f"corpus folder not found: {root}")
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue  # .DS_Store and friends differ between machines
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def current_fingerprint(settings: Settings) -> EvalFingerprint:
    reranking = settings.rerank
    return EvalFingerprint(
        generator_model=settings.llm_model,
        llm_temperature=settings.llm_temperature,
        llm_reasoning_effort=settings.llm_reasoning_effort,
        prompt_version=PROMPT_VERSION,
        judge_model=settings.judge_model,
        citation_prompt_version=CITATION_PROMPT_VERSION,
        embedding_model=settings.embedding_model,
        chunk_strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        rerank=reranking,
        rerank_candidates=settings.rerank_candidates if reranking else 0,
        reranker_model=settings.reranker_model if reranking else "",
        golden_sha256=file_sha256(settings.golden_dataset_path),
        corpus_sha256=tree_sha256(settings.raw_data_dir),
    )


def fingerprint_changes(
    saved: Mapping[str, Any], current: EvalFingerprint
) -> tuple[FieldChange, ...]:
    """Every field that differs; a field missing on either side counts as a change."""
    now = current.to_dict()
    names = [*now, *(name for name in saved if name not in now)]
    return tuple(
        FieldChange(name, saved.get(name, MISSING), now.get(name, MISSING))
        for name in names
        if saved.get(name, MISSING) != now.get(name, MISSING)
    )
