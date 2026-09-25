# Chunking Strategies

Chunking splits long documents into smaller passages before they are embedded. Chunk size and strategy have a large, often underestimated effect on both retrieval and generation quality.

## Fixed-size chunking

Fixed-size chunking cuts text every N characters or tokens, for example every 500 tokens. It is simple and fast, but it ignores structure: it can cut a sentence in half or split a table or a numbered procedure across two chunks.

## Recursive chunking

Recursive chunking tries a list of separators in order, typically paragraph breaks first, then line breaks, then spaces, and only falls back to a hard character limit when a piece is still too large. It keeps paragraphs and sentences intact far more often than fixed-size chunking, at almost no extra cost, which is why it is the most common production default.

## Semantic chunking

Semantic chunking embeds consecutive sentences and starts a new chunk where the similarity between neighbours drops, which signals a topic shift. It produces the most coherent chunks but is slower and more expensive at ingestion time, because every sentence has to be embedded. It is usually reserved for high-value or complex documents.

## Chunk overlap

Overlap repeats the last part of one chunk at the start of the next, commonly 10 to 20 percent of the chunk size. It keeps a fact that sits on a boundary retrievable from at least one chunk. Overlap must always be smaller than the chunk size.

## Choosing a size

Small chunks (around 200 tokens) give precise matches but may lack the context needed to answer. Large chunks (1000 tokens or more) carry more context but dilute the embedding and waste prompt space. The right size should be chosen by measuring retrieval metrics on a golden dataset, not by intuition.
