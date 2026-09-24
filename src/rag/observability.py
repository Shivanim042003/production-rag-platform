import logging
import time
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger("rag")


@dataclass
class StageTrace:
    name: str
    duration_ms: float


@dataclass
class RAGTrace:
    query: str
    stages: list[StageTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_duration_ms: float = 0.0

    def add_stage(
        self,
        name: str,
        duration_ms: float,
    ) -> None:
        self.stages.append(
            StageTrace(
                name=name,
                duration_ms=duration_ms,
            )
        )

    def finish(
        self,
        total_duration_ms: float,
    ) -> None:
        self.total_duration_ms = total_duration_ms


class StageTimer:
    def __init__(
        self,
        trace: RAGTrace,
        stage_name: str,
    ) -> None:
        self.trace = trace
        self.stage_name = stage_name
        self.start_time = 0.0

    def __enter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        duration_ms = (
            time.perf_counter() - self.start_time
        ) * 1000

        self.trace.add_stage(
            name=self.stage_name,
            duration_ms=duration_ms,
        )