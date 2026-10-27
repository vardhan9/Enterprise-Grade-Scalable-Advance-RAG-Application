from typing import List
import logfire

from app.config import settings


def _hard_split(text: str, size: int, overlap: int) -> List[str]:
    """Character-window split for oversized paragraphs; respects overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = max(1, size - overlap)
    parts = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start += step
    return [p for p in parts if p]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """
    Paragraph packer with character overlap between consecutive chunks.

    Packs paragraphs (split on blank lines) until chunk_size, then starts the
    next chunk with the last chunk_overlap characters of the previous chunk.
    Oversized single paragraphs are hard-split with the same overlap.
    """
    size = settings.CHUNK_SIZE if chunk_size is None else int(chunk_size)
    overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else int(chunk_overlap)
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        clamped = max(0, size // 5)
        logfire.warning(
            f"chunk_overlap={overlap} >= chunk_size={size}; clamping overlap to {clamped}"
        )
        overlap = clamped

    with logfire.span(
        "✂️ Text Chunking",
        text_length=len(text),
        chunk_size=size,
        chunk_overlap=overlap,
    ):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current = ""

        def _flush_and_overlap() -> str:
            nonlocal current
            piece = current.strip()
            if piece:
                chunks.append(piece)
            if overlap and piece:
                return piece[-overlap:]
            return ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # Oversized paragraph: flush current, then hard-split p
            if len(p) > size:
                if current.strip():
                    carry = _flush_and_overlap()
                    current = carry
                else:
                    carry = ""
                # Prepend carry into first hard-split window via joining
                to_split = (carry + "\n\n" + p) if carry else p
                hard = _hard_split(to_split, size, overlap)
                if not hard:
                    current = ""
                    continue
                chunks.extend(hard[:-1])
                current = hard[-1]
                continue

            candidate = (current + "\n\n" + p) if current.strip() else p
            if len(candidate) <= size:
                current = candidate
            else:
                carry = _flush_and_overlap()
                current = (carry + "\n\n" + p) if carry else p
                # If carry+p still exceeds size (rare), hard-split
                if len(current) > size:
                    hard = _hard_split(current, size, overlap)
                    chunks.extend(hard[:-1])
                    current = hard[-1] if hard else ""

        if current.strip():
            chunks.append(current.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(
            f"✅ Generated {len(valid_chunks)} chunks "
            f"(chunk_size={size}, chunk_overlap={overlap})"
        )
        return valid_chunks
