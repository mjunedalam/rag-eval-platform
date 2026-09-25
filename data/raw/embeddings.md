# Embeddings

An embedding model maps a piece of text to a dense vector of numbers, for example 384 or 1536 dimensions, so that texts with similar meaning end up close together in vector space.

## Measuring similarity

Similarity between two embeddings is usually measured with **cosine similarity**, the cosine of the angle between the vectors. It ranges from -1 to 1, and higher means more similar. When vectors are normalised to unit length, cosine similarity equals the dot product, which is cheaper to compute.

## One model for documents and queries

Queries and document chunks must be embedded with the **same model**. Vectors produced by different models live in different spaces, so comparing them gives meaningless scores. Changing the embedding model therefore means re-embedding the entire corpus.

## Open-source versus hosted models

- **Sentence Transformers** models such as `all-MiniLM-L6-v2` run locally, cost nothing per call, and produce 384-dimensional vectors. They are a good default for development and CI because they need no API key.
- **Hosted models** such as OpenAI's `text-embedding-3-small` (1536 dimensions) often score higher on retrieval benchmarks but add network latency, a per-token cost, and a dependency on an external provider.

## Limits of dense embeddings

Dense embeddings capture meaning well but can miss exact matches such as product codes, error numbers, or rare names. Hybrid search, which combines dense vectors with keyword search such as BM25, is a common fix.
