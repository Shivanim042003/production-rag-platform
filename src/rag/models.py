from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    document_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
