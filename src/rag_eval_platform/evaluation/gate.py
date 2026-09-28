"""The evaluation gate: decide whether a change may merge.

Retrieval is scored live on every pull request. Generation needs a local GPU, so its scores
come from the committed baseline, which must be fresh: made with the current settings,
golden set and corpus (see fingerprint.py).
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, Self

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.baselines import Baseline
from rag_eval_platform.evaluation.evaluator import RetrievalReport
from rag_eval_platform.evaluation.fingerprint import (
    EvalFingerprint,
    FieldChange,
    fingerprint_changes,
)

GATE_MARKER = "<!-- rag-eval-gate -->"  # lets CI find and update its own PR comment
FIX_HINT = (
    "Fix: run `uv run python scripts/run_generation_evaluation.py --save-baseline` "
    "and commit `baselines/generation.json`."
)
_EPSILON = 1e-9  # 0.93 - 0.91 is 0.020000000000000018 in floating point

CheckStatus = Literal["pass", "fail"]


@dataclass(frozen=True)
class GateLimits:
    min_recall_at_k: float
    min_mrr: float
    min_ndcg_at_k: float
    min_faithfulness: float
    min_answer_relevance: float
    max_drop: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.min_recall_at_k, settings.min_mrr, settings.min_ndcg_at_k,
                   settings.min_faithfulness, settings.min_answer_relevance,
                   settings.gate_max_drop)  # fmt: skip


@dataclass(frozen=True)
class Check:
    group: Literal["retrieval", "generation"]
    name: str
    value: float | None
    threshold: float | None
    baseline: float | None
    status: CheckStatus
    reason: str


@dataclass(frozen=True)
class GateResult:
    checks: tuple[Check, ...]
    changes: tuple[FieldChange, ...]
    passed: bool


def run_gate(
    retrieval: RetrievalReport,
    retrieval_baseline: Baseline | None,
    generation_baseline: Baseline | None,
    current: EvalFingerprint,
    limits: GateLimits,
) -> GateResult:
    saved = _overall(retrieval_baseline)
    checks = [
        _retrieval_check(
            name,
            getattr(retrieval.overall, field),
            getattr(limits, limit),
            _number(saved.get(field)) if retrieval_baseline else None,
            limits.max_drop,
        )
        for name, field, limit in (
            ("recall@k", "recall", "min_recall_at_k"),
            ("mrr", "mrr", "min_mrr"),
            ("ndcg@k", "ndcg", "min_ndcg_at_k"),
        )
    ]
    changes: tuple[FieldChange, ...] = ()
    if generation_baseline is None:
        checks.append(_freshness("fail", f"no generation baseline. {FIX_HINT}"))
    else:
        changes = fingerprint_changes(generation_baseline.fingerprint, current)
        if changes:
            listed = "; ".join(str(change) for change in changes)
            checks.append(_freshness("fail", f"stale ({listed}). {FIX_HINT}"))
        else:
            checks.append(_freshness("pass", "made with the current settings"))
    scores = _overall(generation_baseline)
    for name, limit in (("faithfulness", limits.min_faithfulness),
                        ("answer_relevance", limits.min_answer_relevance)):  # fmt: skip
        checks.append(_generation_check(name, _number(scores.get(name)), limit))
    return GateResult(tuple(checks), changes, all(c.status == "pass" for c in checks))


def gate_markdown(
    result: GateResult, retrieval: RetrievalReport, retrieval_baseline: Baseline | None
) -> str:
    headline = ("## ✅ Evaluation gate passed" if result.passed
                else "## ⛔ Evaluation gate blocked this change")  # fmt: skip
    lines = [GATE_MARKER, headline, "",
             "| Check | Baseline | Now | Change | Minimum | Result |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for c in result.checks:
        change = (f"{c.value - c.baseline:+.3f}"
                  if c.value is not None and c.baseline is not None else "-")  # fmt: skip
        mark = "✅" if c.status == "pass" else "⛔"
        lines.append(f"| {c.name} | {_fmt(c.baseline)} | {_fmt(c.value)} | {change} | "
                     f"{_fmt(c.threshold, 2)} | {mark} {c.reason} |")  # fmt: skip
    lines += ["", "Generation scores come from `baselines/generation.json`, judged locally "
              "(CI has no GPU)."]  # fmt: skip
    if result.changes:
        lines += ["", "### Stale generation baseline", ""]
        lines += [f"- `{change}`" for change in result.changes]
        lines += ["", FIX_HINT]
    lines += ["", "### Retrieval by query type", "",
              "| Query type | Recall before → now | MRR before → now | NDCG before → now |",
              "|---|---|---|---|"]  # fmt: skip
    before = retrieval_baseline.report.get("by_query_type", {}) if retrieval_baseline else {}
    for query_type, scores in sorted(retrieval.by_query_type.items()):
        old = before.get(query_type, {}) if isinstance(before, dict) else {}
        cells = [f"{_fmt(_number(old.get(m)))} → {getattr(scores, m):.3f}"
                 for m in ("recall", "mrr", "ndcg")]  # fmt: skip
        lines.append(f"| {query_type} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def gate_error_markdown(message: str) -> str:
    """The report when the gate could not run; it replaces any earlier PR comment."""
    safe = message.replace("|", "\\|")
    return (f"{GATE_MARKER}\n## ⚠️ Evaluation gate could not run\n\n{safe}\n\n"
            "The check fails until the gate runs again; see the job log.\n")  # fmt: skip


def gate_to_dict(result: GateResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "checks": [asdict(c) for c in result.checks],
        "changes": [asdict(c) for c in result.changes],
    }


def gate_from_dict(data: Mapping[str, Any]) -> GateResult:
    return GateResult(
        tuple(Check(**c) for c in data["checks"]),
        tuple(FieldChange(**c) for c in data["changes"]),
        bool(data["passed"]),
    )


def _retrieval_check(
    name: str, value: float, threshold: float, baseline: float | None, max_drop: float
) -> Check:
    def check(status: CheckStatus, reason: str) -> Check:
        return Check("retrieval", name, value, threshold, baseline, status, reason)

    if value < threshold:
        return check("fail", f"{value:.3f} < {threshold:.2f}")
    if baseline is None:
        return check("pass", f"{value:.3f} >= {threshold:.2f} (no baseline to compare)")
    drop = baseline - value
    if drop > max_drop + _EPSILON:
        return check("fail", f"dropped {drop:.3f} from {baseline:.3f} (max {max_drop:.2f})")
    return check("pass", f"{value:.3f} >= {threshold:.2f}, within {max_drop:.2f} of baseline")


def _freshness(status: CheckStatus, reason: str) -> Check:
    return Check("generation", "freshness", None, None, None, status, reason)


def _generation_check(name: str, value: float | None, threshold: float) -> Check:
    if value is None:
        return Check("generation", name, None, threshold, None, "fail", "never scored")
    status: CheckStatus = "pass" if value >= threshold else "fail"
    comparison = ">=" if status == "pass" else "<"
    return Check("generation", name, value, threshold, None, status,
                 f"{value:.3f} {comparison} {threshold:.2f}")  # fmt: skip


def _overall(baseline: Baseline | None) -> Mapping[str, Any]:
    overall = baseline.report.get("overall", {}) if baseline else {}
    return overall if isinstance(overall, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"
