# Retrieval Metrics

Retrieval metrics answer one question: did the system fetch the right information? They compare the ranked list of retrieved documents with the set of documents known to be relevant, and they are computed at a cut-off k, the number of results considered.

## Precision@k

Precision@k is the number of relevant documents in the top k divided by the number of documents retrieved in the top k. It measures how much noise reaches the prompt.

## Recall@k

Recall@k is the number of relevant documents in the top k divided by the total number of relevant documents. It measures whether the sources needed for the answer were found at all. For RAG, recall is usually the most important retrieval metric, because a passage that is never retrieved cannot be used.

## Mean Reciprocal Rank (MRR)

The reciprocal rank of one query is 1 divided by the rank of the first relevant document, or 0 if no relevant document is retrieved. A relevant document at rank 1 scores 1.0, at rank 2 scores 0.5, and at rank 4 scores 0.25. MRR is the mean of this value over all queries. It measures how high the first relevant document appears in the ranked results.

## NDCG@k

Normalised Discounted Cumulative Gain rewards relevant documents that appear near the top of the list. DCG sums each result's relevance divided by log2(rank + 1). NDCG divides DCG by the ideal DCG, the score of a perfect ranking, so it always falls between 0 and 1. It is most useful when a question has several relevant documents.

## Binary relevance

In the simplest setup every document is either relevant (1) or not (0). Graded relevance, for example 0 to 3, is more expressive but much more expensive to label.
