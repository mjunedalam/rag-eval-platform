# RAG Eval Platform

An end-to-end Retrieval-Augmented Generation (RAG) pipeline where **evaluation is a first-class part of the system**, not an afterthought. Every change to prompts, chunking, embeddings, or retriever config is scored against a golden dataset and gated in CI/CD before it can merge.

> Full design notes: [RAG Pipeline Evaluation – Enterprise Guide](docs/RAG%20Pipeline%20Evaluation%20-%20Enterprise%20Guide.md)

## Three layers of evaluation

| Layer | Question it answers | Metrics |
|---|---|---|
| **Retrieval** | Did we fetch the right information? | Precision, Recall, MRR, NDCG |
| **Generation** | Did we generate the right answer from it? | Faithfulness, Hallucination rate, Answer & Context Relevance |
| **Production observability** | Is it still performing well over time? | Logging, background auto-eval, human review, feedback loops, drift detection |

## Pipeline

```
Query → Embedding → Retrieval (+ re-rank) → Generation (with citations) → Evaluation & Logging
                                                                               │
                                          Golden dataset ← Feedback ← Observability loop
```

## Planned components

- **Ingestion** — loaders, fixed-size / recursive / semantic chunking, embeddings
- **Retrieval** — Chroma or Qdrant, configurable top-k, optional cross-encoder re-ranker
- **Generation** — structured prompts with source attribution
- **Golden dataset** — 50–100 curated question / context / answer triples in `data/golden_dataset/`
- **Evaluation** — RAGAS and DeepEval scoring with pass/fail thresholds (e.g. faithfulness ≥ 0.85)
- **CI/CD gate** — GitHub Actions fails the build when scores regress
- **Observability** — query logging and a Streamlit dashboard of scores over time
- **Enterprise concerns** — cost and latency tracking, PII handling, retrieval-time access control, audit logging

## Tech stack

Python · LangChain / LangGraph · Chroma / Qdrant · OpenAI or Sentence Transformers embeddings · RAGAS · DeepEval · TruLens · Pytest · GitHub Actions · FastAPI · Docker · Streamlit

## Status

Early stage: design docs are in [`docs/`](docs/), and the implementation is in progress.

```bash
git clone git@github.com:mjunedalam/rag-eval-platform.git
```

## Author

**Mohammad Juned Alam**, Senior AI & Software Engineer
[GitHub](https://github.com/TechHorizonsByJuned) · [LinkedIn](https://www.linkedin.com/in/moh-juned-alam-97336416b)

## License

MIT
