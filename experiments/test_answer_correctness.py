from rag.answer_correctness import (
    EmbeddingAnswerCorrectnessScorer,
)


def main() -> None:
    scorer = EmbeddingAnswerCorrectnessScorer()

    examples = [
        (
            "Strong match",
            (
                "PostgreSQL indexes improve query performance "
                "by helping the database locate relevant rows "
                "efficiently."
            ),
            (
                "PostgreSQL indexes are used to improve query "
                "performance by allowing the database to locate "
                "relevant rows more efficiently instead of "
                "scanning the entire table."
            ),
        ),
        (
            "Partial match",
            (
                "PostgreSQL indexes improve query performance."
            ),
            (
                "PostgreSQL indexes are used to improve query "
                "performance by allowing the database to locate "
                "relevant rows more efficiently instead of "
                "scanning the entire table."
            ),
        ),
        (
            "Unrelated answer",
            (
                "PostgreSQL uses MVCC to provide concurrency "
                "control for transactions."
            ),
            (
                "PostgreSQL indexes are used to improve query "
                "performance by allowing the database to locate "
                "relevant rows more efficiently instead of "
                "scanning the entire table."
            ),
        ),
    ]

    print("=" * 70)
    print("ANSWER CORRECTNESS SANITY CHECK")
    print("=" * 70)

    for name, answer, reference_answer in examples:
        result = scorer.score(
            answer=answer,
            reference_answer=reference_answer,
        )

        print()
        print(name)
        print(f"Score: {result.score:.4f}")
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()