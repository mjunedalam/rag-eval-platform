# Generation Metrics

Generation metrics answer a second question: given the retrieved context, did the model produce a correct and grounded answer? Most of them are scored by an LLM acting as a judge.

## Faithfulness

Faithfulness, also called groundedness, measures whether every claim in the answer is supported by the retrieved context. A common method splits the answer into individual claims and checks each one against the context; the score is the share of supported claims. An answer can be correct in the real world and still unfaithful, if its facts came from the model's memory rather than the context.

## Hallucination rate

The hallucination rate is the share of answers that contain at least one unsupported or incorrect claim. It is the operational counterpart of faithfulness.

## Answer relevance

Answer relevance measures whether the answer actually addresses the question that was asked. A fully faithful answer can still be irrelevant, for example when it summarises the context instead of answering.

## Context precision and context recall

Context precision measures whether the retrieved chunks are relevant to the question, with relevant chunks ranked high. Context recall measures whether the retrieved context contains all the information needed to produce the reference answer. They connect retrieval quality to generation quality.

## Frameworks

**RAGAS** is an open-source framework built specifically for RAG evaluation; it implements faithfulness, answer relevance, context precision, and context recall. **DeepEval** offers similar metrics written as pytest-style test cases, which makes it easy to run evaluation inside a test suite. **TruLens** focuses on tracking these scores over time with feedback functions.
