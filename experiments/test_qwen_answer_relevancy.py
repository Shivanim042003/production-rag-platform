from rag.answer_relevancy import QwenAnswerRelevancyScorer


def main() -> None:
    scorer = QwenAnswerRelevancyScorer()

    relevant_result = scorer.score(
        query="What are PostgreSQL indexes used for?",
        answer=(
            "PostgreSQL indexes improve query performance "
            "by helping the database locate rows efficiently."
        ),
    )

    irrelevant_result = scorer.score(
        query="What are PostgreSQL indexes used for?",
        answer=(
            "PostgreSQL uses MVCC to allow multiple "
            "transactions to operate concurrently."
        ),
    )

    partial_result = scorer.score(
        query="What are PostgreSQL indexes used for?",
        answer=(
            "PostgreSQL uses indexes."
        ),
    )

    print("=" * 70)
    print("QWEN ANSWER RELEVANCY SANITY CHECK")
    print("=" * 70)

    print("\nRelevant example")
    print(f"Score: {relevant_result.score}")
    print(f"Relevant: {relevant_result.relevant}")
    print(f"Reason: {relevant_result.reason}")

    print("\nIrrelevant example")
    print(f"Score: {irrelevant_result.score}")
    print(f"Relevant: {irrelevant_result.relevant}")
    print(f"Reason: {irrelevant_result.reason}")

    print("\nPartial example")
    print(f"Score: {partial_result.score}")
    print(f"Relevant: {partial_result.relevant}")
    print(f"Reason: {partial_result.reason}")


if __name__ == "__main__":
    main()