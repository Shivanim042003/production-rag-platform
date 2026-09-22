from rag.qwen_grounding import QwenGroundingScorer


def run_case(
    scorer: QwenGroundingScorer,
    name: str,
    answer: str,
    context: list[str],
) -> None:
    result = scorer.score(
        answer=answer,
        context=context,
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("\nANSWER")
    print(answer)

    print("\nCONTEXT")
    for index, text in enumerate(context, start=1):
        print(f"[Context {index}] {text}")

    print("\nGROUNDING")
    print(f"Grounded: {result.grounded}")
    print(f"Score:    {result.score:.1f}")
    print(f"Reason:   {result.reason}")


def main() -> None:
    scorer = QwenGroundingScorer(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        threshold=0.5,
    )

    run_case(
        scorer=scorer,
        name="SUPPORTED ANSWER",
        answer=(
            "PostgreSQL indexes improve query performance by "
            "allowing PostgreSQL to locate rows without scanning "
            "the entire table."
        ),
        context=[
            (
                "Indexes improve query performance by allowing "
                "PostgreSQL to locate rows without scanning an "
                "entire table."
            )
        ],
    )

    run_case(
        scorer=scorer,
        name="UNSUPPORTED ANSWER",
        answer=(
            "PostgreSQL indexes reduce memory usage by 90 percent."
        ),
        context=[
            (
                "Indexes improve query performance by allowing "
                "PostgreSQL to locate rows without scanning an "
                "entire table."
            )
        ],
    )


if __name__ == "__main__":
    main()
