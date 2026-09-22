from rag.local_generator import LocalGenerator


def main() -> None:
    generator = LocalGenerator(
        model_name="google/flan-t5-base",
        max_new_tokens=128,
    )

    query = "What are PostgreSQL indexes used for?"

    context = [
        (
            "Indexes improve query performance by allowing PostgreSQL "
            "to locate rows without scanning an entire table."
        ),
        (
            "Common PostgreSQL index types include B-tree, Hash, "
            "GiST, SP-GiST, GIN, and BRIN."
        ),
    ]

    result = generator.generate(
        query=query,
        context=context,
    )

    print("=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    print("\n" + "=" * 70)
    print("CONTEXT")
    print("=" * 70)

    for index, text in enumerate(context, start=1):
        print(f"[Context {index}] {text}")

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(result.answer)

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)
    print(result.model)


if __name__ == "__main__":
    main()
