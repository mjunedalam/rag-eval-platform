# Production Observability for RAG

Offline evaluation on a golden dataset shows how a system performed before release. Production observability shows whether it keeps performing well with real users over time.

## Logging

Log every query together with the retrieved chunk ids, the final answer, the latency of each stage, and the token cost. Without these records a bad answer cannot be traced back to its cause.

## Background evaluation

A scheduled job scores a sample of recent production traffic, for example 5 percent of queries, with the same generation metrics used offline, such as faithfulness. There is no reference answer in production, so only reference-free metrics can be used.

## Human review

Answers with low faithfulness or low confidence are routed to a human reviewer. Reviewed cases are the best source of new golden examples.

## Feedback loops

Thumbs-up and thumbs-down buttons and user corrections are collected and triaged. Confirmed failures are added to the golden dataset, so the next evaluation run covers them.

## Drift detection

Quality can decline without any code change. **Query drift** happens when users start asking new kinds of questions. **Knowledge-base drift** happens when documents are added, removed, or go out of date. Watching retrieval scores and the distribution of query types over time reveals both.

## Dashboards

A small dashboard, for example built with Streamlit, can show evaluation scores over time, recent queries, and latency, which is enough to spot a regression early.
