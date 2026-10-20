from rag.answer_relevancy import EmbeddingAnswerRelevancyScorer


def main() -> None:
    scorer = EmbeddingAnswerRelevancyScorer()

    examples = [
        (
            "Relevant",
            "PostgreSQL indexes improve query performance "
            "by helping the database locate rows efficiently.",
        ),
        (
            "Irrelevant",
            "PostgreSQL uses MVCC to allow multiple "
            "transactions to operate concurrently.",
        ),
        (
            "Partial",
            "PostgreSQL uses indexes.",
        ),
    ]

    query = "What are PostgreSQL indexes used for?"

    print("=" * 70)
    print("EMBEDDING ANSWER RELEVANCY SANITY CHECK")
    print("=" * 70)

    for label, answer in examples:
        result = scorer.score(query=query, answer=answer)

        print(f"\n{label} example")
        print(f"Answer: {answer}")
        print(f"Score: {result.score}")
        print(f"Relevant: {result.relevant}")
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()
