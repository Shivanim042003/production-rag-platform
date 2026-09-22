from pathlib import Path

from rag.chunking import fixed_size_chunks
from rag.cleaning import clean_text
from rag.models import DocumentChunk, RawDocument


class TxtLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RawDocument:
        if self.path.suffix.lower() != ".txt":
            raise ValueError(f"Unsupported file type: {self.path.suffix}")

        if not self.path.exists():
            raise FileNotFoundError(f"Document not found: {self.path}")

        if not self.path.is_file():
            raise ValueError(f"Path is not a file: {self.path}")

        text = self.path.read_text(encoding="utf-8")
        cleaned_text = clean_text(text)

        return RawDocument(
            document_id=self.path.stem,
            text=cleaned_text,
            source=str(self.path),
            metadata={
                "file_type": "txt",
            },
        )


def ingest_txt(
    path: str | Path,
    chunk_size: int,
    overlap: int = 0,
) -> list[DocumentChunk]:
    loader = TxtLoader(path)
    document = loader.load()

    return fixed_size_chunks(
        document,
        chunk_size=chunk_size,
        overlap=overlap,
    )
