from rag.multi_query import QwenMultiQueryGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

device = torch.device("cpu")
model.to(device)

generator = QwenMultiQueryGenerator(
    tokenizer=tokenizer,
    model=model,
    device=device,
    model_name=MODEL_NAME,
    num_queries=3,
)

query = "What are PostgreSQL indexes used for?"

print("\nOriginal query:")
print(query)

queries = generator.generate(query)

print("\nGenerated queries:")
for i, generated_query in enumerate(queries, start=1):
    print(f"{i}. {generated_query}")
