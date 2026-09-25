# Golden Datasets

A golden dataset is a curated set of questions with known correct answers and known relevant sources. It is the ground truth that every evaluation run is compared against.

## What each example contains

A typical example holds a stable identifier, the question as a user would phrase it, a reference answer written by a domain expert, the documents that contain the answer, and a query type used to break scores down.

## Size

Start with 50 to 100 carefully reviewed examples. A small, trustworthy set is more useful than a large, noisy one, because every wrong label shows up as a false regression or hides a real one.

## Coverage

The dataset should cover every kind of query users actually ask, not only easy lookups. Typical query types include short factual questions, paraphrased questions that use different words from the source, multi-hop questions whose answer needs two or more documents, and ambiguous questions. Reporting scores per query type exposes a weak slice that an overall average would hide.

## Document-level relevance

Labelling relevance at the document level, rather than the chunk level, keeps the dataset valid when the chunking strategy changes. Retrieved chunks are mapped back to their source document before scoring. The trade-off is that a chunk from the right document but the wrong section still counts as a hit.

## Versioning and growth

Keep the dataset in version control next to the code, so every score can be reproduced and changes to it are reviewed. Grow it over time from production failures and user feedback, because a static dataset can be overfit: the pipeline improves on the benchmark without helping real users.
