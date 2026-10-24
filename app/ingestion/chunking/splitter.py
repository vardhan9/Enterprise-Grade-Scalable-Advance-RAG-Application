from typing import List
import logfire

from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed chunk_size (from settings.CHUNK_SIZE by default).

    chunk_overlap is accepted for experiment config parity; the paragraph packer
    does not implement sliding-window overlap (effective overlap = 0).
    """
    size = settings.CHUNK_SIZE if chunk_size is None else int(chunk_size)
    overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else int(chunk_overlap)
    if overlap:
        logfire.warning(
            f"chunk_overlap={overlap} requested but paragraph splitter ignores overlap "
            "(effective overlap=0)."
        )

    with logfire.span("✂️ Text Chunking", text_length=len(text), chunk_size=size):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) < size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks (chunk_size={size})")
        return valid_chunks
