from rag.models import DocumentChunk, RawDocument


def fixed_size_chunks(
    document: RawDocument,
    chunk_size: int,
    overlap: int = 0,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = document.text.split()

    if not words:
        return []

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            continue

        chunk_index = len(chunks) + 1

        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}_chunk_{chunk_index:03d}",
                document_id=document.document_id,
                text=" ".join(chunk_words),
                source=document.source,
                metadata=document.metadata.copy(),
            )
        )

    return chunks
