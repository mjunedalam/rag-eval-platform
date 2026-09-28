"""Committed evaluation baselines: the reviewed scores the gate compares against.

Unlike ``reports/`` (git-ignored scratch output), ``baselines/`` is tracked: updating a
baseline shows up in the pull request's diff, like changing a threshold.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_eval_platform.evaluation.fingerprint import EvalFingerprint, fingerprint_changes

BASELINE_DIR = Path("baselines")
RETRIEVAL_BASELINE = BASELINE_DIR / "retrieval.json"
GENERATION_BASELINE = BASELINE_DIR / "generation.json"

# Report fields that must match the current settings before a report is adopted.
_NO_FINGERPRINT = (
    "the report has no fingerprint, so its settings are unknown; re-run the evaluation, "
    "or pass --legacy if you are sure it used the current settings"
)
_ADOPT_FIELDS = ("generator_model", "prompt_version", "judge_model", "citation_prompt_version")


class BaselineError(Exception):
    """A baseline file exists but cannot be read."""


@dataclass(frozen=True)
class Baseline:
    report: dict[str, Any]
    fingerprint: dict[str, Any]  # raw, so fields an older version lacked read as changed


def load_baseline(path: Path) -> Baseline | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        report, fingerprint = data["report"], data["fingerprint"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BaselineError(f"{path} is not a valid baseline file: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(fingerprint, dict):
        raise BaselineError(f"{path} is not a valid baseline file: report and fingerprint "
                            "must be objects")  # fmt: skip
    return Baseline(report, fingerprint)


def save_baseline(path: Path, report: Mapping[str, Any], fingerprint: EvalFingerprint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"fingerprint": fingerprint.to_dict(), "report": dict(report)}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def check_adoptable(
    report: Mapping[str, Any],
    fingerprint: EvalFingerprint,
    golden_size: int,
    *,
    legacy: bool = False,
) -> tuple[str, ...]:
    """Why an existing generation report cannot become the baseline (empty means it can).

    A report carries the fingerprint it was made with, which must match exactly. A report
    from before fingerprints existed is only adopted with ``legacy``, trusting that it was
    made with the current retrieval settings; then its recorded versions must match.
    """
    saved = report.get("fingerprint")
    if isinstance(saved, dict):
        problems = [f"the report was made with {change}"
                    for change in fingerprint_changes(saved, fingerprint)]  # fmt: skip
    elif legacy:
        problems = [
            f"{name} is {report.get(name)!r} in the report but {getattr(fingerprint, name)!r} now"
            for name in _ADOPT_FIELDS
            if report.get(name) != getattr(fingerprint, name)
        ]
    else:
        return (_NO_FINGERPRINT,)
    scored = len(report.get("examples", []))
    if scored != golden_size:
        problems.append(f"the report scored {scored} of {golden_size} golden questions")
    return tuple(problems)
