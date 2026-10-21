
from rag.generation_prompt import build_generation_prompt
from rag.generator import GenerationResult
from rag.qwen_model import QwenModelBundle


class QwenLocalGenerator:
    """Generate RAG answers using a local Qwen model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 128,
        model_bundle: QwenModelBundle | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.model_bundle = (
            model_bundle
            if model_bundle is not None
            else QwenModelBundle(
                model_name=model_name,
                device=device,
            )
        )

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> GenerationResult:
        prompt = build_generation_prompt(
            query=query,
            context=context,
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
                "content": prompt,
            },
        ]

        answer = self.model_bundle.generate_tokens(
            messages,
            max_new_tokens=self.max_new_tokens,
        )

        if not answer:
            raise RuntimeError(
                "Qwen model returned an empty answer."
            )

        return GenerationResult(
            answer=answer,
            model=self.model_name,
        )
