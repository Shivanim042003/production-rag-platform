from rag.answer_relevancy import EmbeddingAnswerRelevancyScorer


def main() -> None:
    scorer = EmbeddingAnswerRelevancyScorer()

    examples = [
        (
            "Relevant example",
            "What are PostgreSQL indexes used for?",
            (
                "PostgreSQL indexes improve query performance "
                "by helping the database locate relevant rows efficiently."
            ),
        ),
        (
            "Irrelevant example",
            "What are PostgreSQL indexes used for?",
            (
                "PostgreSQL uses MVCC to provide concurrency "
                "control for transactions."
            ),
        ),
        (
            "Partial example",
            "What are PostgreSQL indexes used for?",
            "PostgreSQL uses indexes.",
        ),
    ]

    print("=" * 70)
    print("EMBEDDING ANSWER RELEVANCY SANITY CHECK")
    print("=" * 70)

    for name, query, answer in examples:
        result = scorer.score(
            query=query,
            answer=answer,
        )

        print()
        print(name)
        print(f"Score: {result.score}")
        print(f"Relevant: {result.relevant}")
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()