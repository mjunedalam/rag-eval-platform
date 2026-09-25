# Cost and Latency

A RAG system can be accurate and still fail in production if it is too slow or too expensive to run at scale.

## Cost

Embedding calls and generation calls are both billed per token. Generation usually dominates, and its cost grows with the number of retrieved chunks placed in the prompt. Teams track **cost per query** and cost per session as core metrics. Common savings are retrieving fewer but better chunks (often with re-ranking), using a smaller model for simple queries, and caching answers to repeated questions.

## Latency

Every stage adds delay: query embedding, vector search, re-ranking, and generation. Generation is typically the slowest stage because the answer is produced token by token. Measure latency **per stage** so it is clear where the time goes.

## Latency budgets and percentiles

A latency budget sets a target such as "under 3 seconds end to end". It should be checked at the **95th percentile (p95)**, not the average, because the average hides the slow requests that users actually notice. Test the budget under realistic concurrent load, not only with single requests.

## Streaming

Streaming the answer token by token lowers the time to first token, which makes the system feel faster even when total generation time is unchanged.
