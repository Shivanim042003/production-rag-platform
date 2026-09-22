from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class GenerationResult:
    answer: str
    model: str


class Generator(Protocol):
    def generate(
        self,
        query: str,
        context: Sequence[str],
    ) -> GenerationResult:
        ...
