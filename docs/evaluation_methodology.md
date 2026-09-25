# Evaluation Methodology

> **Status:** design. This describes how the platform will measure and gate quality. The evaluation modules are placeholders and will be implemented next.

Quality is measured in three layers:

| Layer | Question it answers | When it runs |
|---|---|---|
| Retrieval | Did we fetch the right information? | Every pull request (CI gate) |
| Generation | Did we produce a correct, grounded answer from it? | Every pull request (CI gate) and on sampled production traffic |
| Production observability | Is it still performing well over time, with real users? | Continuously |

## 1. Golden dataset

The golden dataset is the ground truth for evaluation. It lives in `data/golden_dataset/qa_pairs.json` and is versioned with the code, so every score can be reproduced.

**Planned format**, one object per example:

```json
{
  "id": "retrieval-001",
  "question": "What does Mean Reciprocal Rank measure?",
  "expected_answer": "How high the first relevant document appears in the ranked results.",
  "relevant_doc_ids": ["retrieval_metrics.md"],
  "query_type": "short"
}
```

| Field | Meaning |
|---|---|
| `id` | Unique, stable identifier. |
| `question` | The query as a user would type it. |
| `expected_answer` | Reference answer written by a domain expert. |
| `relevant_doc_ids` | Source documents that contain the answer. |
| `query_type` | Slice used to break down scores: `short`, `paraphrase` (different wording from the source), `multi_hop` (needs two or more documents), `ambiguous` |

The current corpus (`data/raw/`, 14 Markdown documents on RAG and LLM evaluation) and its 30 golden examples are a starter set; `evaluation/golden_dataset.py` validates the file and checks that every `relevant_doc_ids` entry exists in the corpus.

**Curation rules**

- Target 50–100 examples to start, reviewed by someone who knows the domain.
- Cover every query type, not only easy questions.
- Relevance is judged at the **document** level, so the dataset stays valid when chunking changes.
- Add new examples from production failures and user feedback (see section 5).

## 2. Retrieval metrics

Computed per question at cut-off *k* (the configured top-k), then averaged. Retrieved chunks are mapped to their source document and de-duplicated in rank order before scoring.

| Metric | Definition | What it tells you |
|---|---|---|
| **Precision@k** | relevant retrieved ÷ retrieved (in the top k) | How much noise reaches the prompt. |
| **Recall@k** | relevant retrieved ÷ all relevant | Whether the answer's sources were found at all. |
| **MRR** | 1 ÷ rank of the first relevant document (0 if none) | Whether the right source is near the top, where the LLM pays attention. |
| **NDCG@k** | DCG ÷ ideal DCG, with DCG = Σ rel_i ÷ log₂(i + 1) | Ranking quality when several documents are relevant. |

Scores are also reported **per `query_type`**, since an average can hide a weak slice (for example multi-hop questions).

## 3. Generation metrics

Scored with RAGAS as the primary framework and DeepEval for pytest-style checks, using an LLM-as-a-judge.

| Metric | Question |
|---|---|
| **Faithfulness / groundedness** | Is every claim in the answer supported by the retrieved context? |
| **Hallucination rate** | Share of answers containing unsupported or incorrect claims. |
| **Answer relevance** | Does the answer address the question that was asked? |
| **Context precision / recall** | Was the retrieved context relevant, and did it contain what the reference answer needs? |
| **Citation validity** | Do the `[n]` markers point to chunks that actually support the claim? |

**Judge setup**

- Use a stronger or different model than the one being evaluated, to limit self-preference bias.
- Pin the judge model and prompt version; changing either means re-establishing the baseline.
- Spot-check a sample of judge verdicts by hand whenever the judge changes.

## 4. CI/CD evaluation gate

`.github/workflows/evaluation_gate.yml` runs on every pull request:

1. Install the project and build the index from `data/raw/`.
2. Run every golden example through the pipeline.
3. Compute retrieval and generation metrics.
4. Compare the averages against the thresholds.
5. Fail the check, and block the merge, if any metric is below its threshold. The full report is uploaded as a build artifact.

**Starting thresholds** (tune once a baseline exists):

| Metric | Minimum |
|---|---|
| Recall@k | 0.80 |
| MRR | 0.70 |
| NDCG@k | 0.70 |
| Faithfulness | 0.85 |
| Answer relevance | 0.80 |

Thresholds live in configuration, not in code. Raising a threshold is a deliberate, reviewed change.

**Keeping CI cheap and stable**

- Retrieval metrics are deterministic and run on every pull request.
- Generation metrics call an LLM, so they cost money and vary slightly between runs. Run them on a fixed sample with a pinned judge, and allow a small tolerance.

## 5. Production observability

| Practice | How |
|---|---|
| **Logging** | Every query, retrieved chunk ids, answer, latency and token cost. |
| **Background evaluation** | Scheduled job scores a sample of recent traffic with the generation metrics. |
| **Human review** | Low-confidence or low-faithfulness answers are routed to a reviewer. |
| **Feedback loop** | Thumbs up/down and corrections become candidate golden examples. |
| **Drift detection** | Watch for shifts in query types or in the knowledge base that lower retrieval scores over time. |
| **Cost and latency** | Track cost per query and p95 latency against a budget (for example under 3 s end to end). |

The Streamlit dashboard (`observability/dashboard.py`) shows evaluation scores over time, recent queries and latency.

## 6. Experiments

Every retrieval or prompt change (chunk size, chunking strategy, top-k, embedding model, re-ranking, prompt wording) is run against the golden dataset. The result is recorded as a before/after table in the pull request, so decisions are backed by numbers.

## Known limitations

- An LLM judge is itself a model and can be wrong; it approximates human judgement rather than replacing it.
- A small golden dataset can overfit: pipeline changes may improve the benchmark without helping real users. Keep growing it from production traffic.
- Document-level relevance does not detect a retrieved chunk that comes from the right document but the wrong section.
