import json

with open("benchmarks/relevance.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} benchmark queries")

for i, item in enumerate(data, start=1):
    print(
        f"{i}. {item['query']} -> "
        f"{len(item['relevant_chunks'])} relevant chunks"
    )
