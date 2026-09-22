import numpy as np

from rag import cross_encoder


class FakeCrossEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.received_pairs = None

    def predict(self, pairs):
        self.received_pairs = pairs
        return np.array([0.8, -0.4, 1.2])


def test_cross_encoder_scorer_loads_model(monkeypatch) -> None:
    monkeypatch.setattr(cross_encoder, "CrossEncoder", FakeCrossEncoder)

    scorer = cross_encoder.CrossEncoderScorer(
        model_name="test-model"
    )

    assert scorer.model_name == "test-model"
    assert scorer.model.model_name == "test-model"


def test_cross_encoder_scorer_predicts_scores(monkeypatch) -> None:
    monkeypatch.setattr(cross_encoder, "CrossEncoder", FakeCrossEncoder)

    scorer = cross_encoder.CrossEncoderScorer()

    pairs = [
        ["PostgreSQL indexes", "PostgreSQL supports indexes."],
        ["Redis caching", "Redis is used for caching."],
        ["database", "A relational database stores data."],
    ]

    scores = scorer.predict(pairs)

    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float32
    assert np.allclose(scores, [0.8, -0.4, 1.2])
    assert scorer.model.received_pairs == pairs


def test_cross_encoder_scorer_returns_empty_array_for_empty_pairs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cross_encoder, "CrossEncoder", FakeCrossEncoder)

    scorer = cross_encoder.CrossEncoderScorer()

    scores = scorer.predict([])

    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float32
    assert scores.shape == (0,)


def test_cross_encoder_scorer_converts_model_output_to_numpy(
    monkeypatch,
) -> None:
    class ListOutputCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def predict(self, pairs):
            return [1, 2, 3]

    monkeypatch.setattr(
        cross_encoder,
        "CrossEncoder",
        ListOutputCrossEncoder,
    )

    scorer = cross_encoder.CrossEncoderScorer()

    scores = scorer.predict(
        [
            ["q1", "d1"],
            ["q2", "d2"],
            ["q3", "d3"],
        ]
    )

    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float32
    assert scores.tolist() == [1.0, 2.0, 3.0]


def test_cross_encoder_scorer_preserves_pair_order(monkeypatch) -> None:
    monkeypatch.setattr(cross_encoder, "CrossEncoder", FakeCrossEncoder)

    scorer = cross_encoder.CrossEncoderScorer()

    pairs = [
        ["query A", "document A"],
        ["query B", "document B"],
        ["query C", "document C"],
    ]

    scorer.predict(pairs)

    assert scorer.model.received_pairs == pairs
