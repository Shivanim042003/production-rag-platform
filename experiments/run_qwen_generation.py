import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.eval()

    query = "What are PostgreSQL indexes used for?"

    context = (
        "Indexes improve query performance by allowing PostgreSQL "
        "to locate rows without scanning an entire table.\n\n"
        "Common PostgreSQL index types include B-tree, Hash, "
        "GiST, SP-GiST, GIN, and BRIN."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document question-answering assistant. "
                "Answer only using the provided context. "
                "Do not use outside knowledge. "
                "Write a complete, direct answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question:\n{query}"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)
    print(MODEL_NAME)


if __name__ == "__main__":
    main()
