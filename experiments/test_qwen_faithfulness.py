from rag.faithfulness import QwenFaithfulnessScorer


def main() -> None:
    scorer = QwenFaithfulnessScorer()

    supported_answer = (
        "PostgreSQL indexes improve query performance."
    )

    supported_context = [
        (
            "PostgreSQL indexes improve query performance "
            "by allowing efficient lookup of rows."
        )
    ]

    unsupported_answer = (
        "PostgreSQL indexes automatically reduce "
        "database storage usage."
    )

    unsupported_context = [
        "PostgreSQL indexes improve query performance."
    ]

    supported_score = scorer.score(
        answer=supported_answer,
        context=supported_context,
    )

    unsupported_score = scorer.score(
        answer=unsupported_answer,
        context=unsupported_context,
    )

    print("Supported example:")
    print(f"Score: {supported_score}")

    print()

    print("Unsupported example:")
    print(f"Score: {unsupported_score}")


if __name__ == "__main__":
    main()