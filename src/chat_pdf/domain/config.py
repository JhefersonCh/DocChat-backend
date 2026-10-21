from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    """Pipeline knobs. Code-level defaults; sizes are chars (~4 chars per token)."""

    chunk_size: int = 1400  # ~350 tokens; chosen with scripts/eval_retrieval.py (600-1400 tried)
    chunk_overlap: int = 210  # 15%
    search_candidates: int = 20  # per ranking (vector / full-text) before fusion and rerank
    # RRF constant. Lower than the usual 60: with <= 20 candidates per list, k=60 flattens the
    # vector ranks so much that any full-text hit reorders them (measured: worse on paraphrases).
    rrf_k: int = 10
    # RRF weight of the full-text ranking (vector = 1): enough to win exact-term queries, not to
    # override the embedding on paraphrased questions.
    text_weight: float = 0.3
    # RRF weights of the native image / audio rankings (text vector = 1), and when the planner's
    # `modality` hint favors one of them. Rank-based: no cosine is compared across modalities.
    native_weight: float = 0.7
    native_hint_weight: float = 1.5
    embed_batch: int = 32  # texts per embeddings request (applied by the Embedder adapter)
    history_limit: int = 6  # previous messages sent to the model
    snippet_chars: int = 200  # length of the snippet shown in a citation
    max_error_chars: int = 500  # truncation of the error stored on a failed document
    # Best-source cosine similarity under which an answer is flagged as weakly supported.
    # Calibrated live with EmbeddingGemma (top-1 cosine of /search): relevant queries 0.62-0.77
    # (El Principito 0.76-0.77, business plans 0.62-0.72; 0.59 for a topic the plan barely
    # covers); unrelated queries 0.47-0.60 (Principito <=0.57, business plans <=0.60). The gap is
    # narrow on short domain docs, so 0.60 is a soft warning. Retune when changing the model.
    weak_evidence_threshold: float = 0.6
    max_context_chars: int = 24000  # sources sent to the model per answer (~6000 tokens)

    def __post_init__(self) -> None:
        # The chunking loop only advances if the overlap is smaller than the shortest chunk.
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size // 2:
            raise ValueError("chunk_overlap debe estar en [0, chunk_size // 2) y chunk_size > 0")
        if self.search_candidates <= 0 or self.rrf_k <= 0 or self.text_weight < 0:
            raise ValueError("search_candidates y rrf_k deben ser positivos y text_weight >= 0")
        if self.native_weight < 0 or self.native_hint_weight < 0:
            raise ValueError("native_weight y native_hint_weight deben ser >= 0")
        if not 0 <= self.weak_evidence_threshold <= 1:
            raise ValueError("weak_evidence_threshold debe estar en [0, 1]")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars debe ser positivo")
