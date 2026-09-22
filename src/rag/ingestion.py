from pathlib import Path

from rag.models import RawDocument


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

        return RawDocument(
            document_id=self.path.stem,
            text=text,
            source=str(self.path),
            metadata={
                "file_type": "txt",
            },
        )
