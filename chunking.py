from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One searchable section of extracted text."""

    index: int
    text: str
    start_char: int
    end_char: int

def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 400) -> list[Chunk]:
    """Split text into ordered, overlapping chunks."""
    
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]

        if piece.strip():
            chunks.append(
                Chunk(
                    index=index,
                    text=piece,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1

        if end == len(text):
            break

        start = end - overlap

    return chunks