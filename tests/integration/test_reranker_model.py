"""Integration test that loads a real cross-encoder (needs --extra local-embeddings)."""

import pytest

from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker
from rag_eval_platform.retrieval.vector_store import SearchResult

pytestmark = pytest.mark.integration


def test_real_cross_encoder_puts_the_answering_passage_first() -> None:
    pytest.importorskip("sentence_transformers")
    reranker = CrossEncoderReranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    passages = [
        "Photosynthesis converts sunlight into chemical energy.",
        "Newton's second law states that force equals mass times acceleration.",
        "The Eiffel Tower is in Paris.",
    ]
    candidates = [
        SearchResult(Chunk(id=f"d{i}#0", doc_id=f"d{i}", index=0, text=t, start_index=0), 0.0)
        for i, t in enumerate(passages)
    ]

    reranked = reranker.rerank("What does Newton's second law say?", candidates, top_n=2)

    assert reranked[0].chunk.doc_id == "d1"
    assert len(reranked) == 2
