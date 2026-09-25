# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is a pattern in which a language model answers a question using documents fetched at query time, instead of relying only on what it memorised during training. It was introduced by Lewis et al. at Facebook AI Research in 2020.

## Why teams use RAG

- **Fresh knowledge**: the document store can be updated daily without retraining or fine-tuning the model.
- **Private data**: internal documents never have to be part of a model's training set.
- **Traceability**: because the answer is built from specific passages, the system can cite its sources.
- **Lower hallucination**: grounding the model in retrieved text reduces, but does not eliminate, invented facts.

## The two stages

A RAG system has an offline stage and an online stage.

The **offline (ingestion) stage** loads source documents, splits them into chunks, converts each chunk into an embedding vector, and stores the vectors in a vector database.

The **online (query) stage** embeds the user's question, retrieves the most similar chunks, optionally re-ranks them, and passes them to the language model inside a prompt. The model writes an answer from that context.

## Where RAG fails

Most RAG failures fall into two groups. Retrieval failures happen when the right passage is never fetched, so even a perfect model cannot answer. Generation failures happen when the right passage is fetched but the model ignores it, misreads it, or adds claims that are not in it. Evaluating the two stages separately is the only way to know which one to fix.
