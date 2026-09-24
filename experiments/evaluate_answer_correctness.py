from pathlib import Path

from rag.answer_correctness import (
    EmbeddingAnswerCorrectnessScorer,
)
from rag.rag_evaluation_dataset import (
    load_rag_evaluation_dataset,
)


BENCHMARK_DIR = Path("benchmarks")


def main() -> None:
    cases = load_rag_evaluation_dataset(
        BENCHMARK_DIR / "rag_evaluation.json"
    )

    scorer = EmbeddingAnswerCorrectnessScorer()

    print("=" * 70)
    print("ANSWER CORRECTNESS BENCHMARK")
    print("=" * 70)

    total_score = 0.0

    for index, case in enumerate(cases, start=1):
        # Upper-bound sanity check:
        # compare the reference answer with itself.
        result = scorer.score(
            answer=case.reference_answer,
            reference_answer=case.reference_answer,
        )

        total_score += result.score

        print()
        print(f"Case {index}")
        print(f"Query: {case.query}")
        print(f"Score: {result.score:.4f}")

    average_score = total_score / len(cases)

    print()
    print("=" * 70)
    print(f"Cases evaluated: {len(cases)}")
    print(f"Average score:   {average_score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()