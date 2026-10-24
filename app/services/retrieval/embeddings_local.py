import logfire

from app.config import settings

_EMBEDDING_DIM = 384  # all-MiniLM-L12-v2

_active_model = None


# ── Model initialisation ───────────────────────────────────────────────────────

def _load_model():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers model (all-MiniLM-L12-v2, 384-dim).")
    return SentenceTransformer("all-MiniLM-L12-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model
    if _active_model is not None:
        return
    _active_model = _load_model()


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model."""
    return _EMBEDDING_DIM


# ── Batch embedding ─────────────────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """
    Embed already-chunked texts. Text chunk_size is controlled upstream
    (settings.CHUNK_SIZE → chunk_text → embed_texts).
    batch_size controls how many chunks are encoded per encode() call.
    """
    _init()
    size = settings.EMBEDDING_BATCH_SIZE if batch_size is None else int(batch_size)
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), size):
        batch = texts[i : i + size]
        with logfire.span(
            "Embed batch",
            model="local",
            start=i,
            size=len(batch),
            text_chunk_size=settings.CHUNK_SIZE,
        ):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings