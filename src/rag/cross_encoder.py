from collections.abc import Sequence

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderScorer:
    """Scores query-document pairs using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def predict(
        self,
        pairs: Sequence[Sequence[str]],
    ) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=np.float32)

        return np.asarray(
            self.model.predict(list(pairs)),
            dtype=np.float32,
        )
