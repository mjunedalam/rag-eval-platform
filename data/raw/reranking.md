# Re-ranking

Re-ranking is an optional second retrieval step that reorders candidate chunks using a more accurate but slower model.

## Bi-encoders and cross-encoders

The embedding model used for vector search is a **bi-encoder**: it encodes the query and each document separately, so document vectors can be computed once, ahead of time. This is fast but loses fine-grained interaction between query and document words.

A **cross-encoder** reads the query and one candidate document together in a single pass and outputs a relevance score. Because it sees both texts at once, it judges relevance more accurately, but it must run once per query-document pair at query time, so it cannot be used to search a whole corpus.

## Retrieve then re-rank

The standard pattern combines both. The vector store first retrieves a larger candidate set, for example the top 20 or 50 chunks, and the cross-encoder then scores each candidate and keeps the best few, for example the top 5, for the prompt.

## Common models

Cross-encoders trained on the MS MARCO passage ranking dataset, such as `cross-encoder/ms-marco-MiniLM-L-6-v2`, are a common open-source choice. Hosted re-ranking APIs are also available.

## Cost

Re-ranking adds latency that grows with the number of candidates, typically tens to a few hundred milliseconds. It mainly improves ranking-sensitive metrics such as MRR and NDCG. Whether it is worth the latency should be decided by measuring those metrics with and without it.
