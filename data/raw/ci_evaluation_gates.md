# Evaluation Gates in CI/CD

An evaluation gate runs the golden dataset through the pipeline on every pull request and fails the build when quality drops, so a regression is caught before it merges rather than after users notice.

## How a gate works

1. Build the index from the source documents.
2. Run every golden question through the pipeline.
3. Compute retrieval and generation metrics.
4. Compare each average against a minimum threshold.
5. Fail the check, and block the merge, if any metric is below its threshold. Publish the full report as a build artifact.

## Thresholds

Typical starting thresholds are a recall@k of 0.80, an MRR of 0.70, and a faithfulness of 0.85. Thresholds belong in configuration, not in code, and raising one should be a deliberate, reviewed change. Thresholds should be tuned once a baseline has been measured.

## Keeping the gate cheap and stable

Retrieval metrics are deterministic and free to compute, so they can run on every pull request. Generation metrics call an LLM judge: they cost money and vary slightly between runs. A common approach is to score generation on a fixed sample with a pinned judge and allow a small tolerance before failing.

## Separate from unit tests

Keep the evaluation gate in its own workflow, separate from lint and unit tests. The two answer different questions: unit tests check that the code behaves as written, while the gate checks that the system still answers well.

## Recording experiments

Every change to chunking, embeddings, top-k, re-ranking, or prompts should include a before-and-after metrics table in the pull request, so decisions are backed by numbers.
