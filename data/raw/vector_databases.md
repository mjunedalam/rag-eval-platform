# Vector Databases

A vector database stores embeddings and returns the vectors nearest to a query vector.

## Approximate nearest neighbour search

Comparing a query against every stored vector (exact search) becomes too slow for large collections. Vector databases use **approximate nearest neighbour (ANN)** indexes instead. The most widely used index is **HNSW** (Hierarchical Navigable Small World), a layered graph that finds near neighbours in roughly logarithmic time while trading a small amount of accuracy for a large gain in speed.

## Chroma

Chroma is an open-source vector database that can run embedded inside a Python process and persist to a local folder. It needs no separate server, which makes it a good fit for local development, notebooks, and tests.

## Qdrant

Qdrant is an open-source vector database written in Rust that runs as a separate service, locally in Docker or as a managed cloud offering. It supports rich payload (metadata) filtering, horizontal scaling, and snapshots, which makes it suited to production.

## Metadata filtering

Each vector can carry metadata such as source document, department, or access level. A filter applied **during** the search, for example "only documents this user may read", guarantees that every returned result satisfies it. Filtering **after** retrieval can silently return fewer than top-k results, and it risks exposing restricted content if the post-filter step is skipped.

## Top-k

The top-k parameter sets how many nearest chunks the search returns. A higher k raises recall but adds noise and prompt tokens; a lower k is precise but may miss the passage that contains the answer.
