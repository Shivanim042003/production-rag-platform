
import re

from rag.grounding import GroundingResult
from rag.qwen_model import QwenModelBundle


class QwenGroundingScorer:
    """Check whether an answer is supported by its context."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        threshold: float = 0.5,
        model_bundle: QwenModelBundle | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0."
            )

        self.model_name = model_name
        self.threshold = threshold

        self.model_bundle = (
            model_bundle
            if model_bundle is not None
            else QwenModelBundle(
                model_name=model_name,
                device=device,
            )
        )

    def score(
        self,
        answer: str,
        context: list[str],
    ) -> GroundingResult:
        if not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not context:
            raise ValueError("Context cannot be empty.")

        formatted_context = "\n\n".join(
            f"[Context {index}]\n{text}"
            for index, text in enumerate(context, start=1)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict answer-grounding evaluator.\n"
                    "Determine whether the answer is fully supported "
                    "by the provided context.\n"
                    "Do not use outside knowledge.\n"
                    "Reply NO if any statement in the answer is missing "
                    "from the context or combines facts the context "
                    "does not connect.\n\n"
                    "Reply with exactly one word:\n"
                    "YES\n"
                    "or\n"
                    "NO"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<CONTEXT>\n{formatted_context}\n</CONTEXT>\n\n"
                    f"<ANSWER>\n{answer}\n</ANSWER>"
                ),
            },
        ]

        decision = self.model_bundle.generate_tokens(
            messages,
            max_new_tokens=8,
        )

        match = re.search(
            r"\b(YES|NO)\b",
            decision.upper(),
        )

        if match is None:
            raise RuntimeError(
                "Qwen returned an invalid grounding decision: "
                f"{decision!r}"
            )

        grounded = match.group(1) == "YES"

        return GroundingResult(
            grounded=grounded,
            score=1.0 if grounded else 0.0,
            reason=(
                "Qwen judged the answer to be supported "
                "by the context."
                if grounded
                else "Qwen judged the answer to be unsupported "
                "by the context."
            ),
        )
