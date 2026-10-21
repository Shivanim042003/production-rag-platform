from collections.abc import Callable
from threading import Event, Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


class _StopWhenSet(StoppingCriteria):
    """Stop generation once the given event is set."""

    def __init__(self, event: Event) -> None:
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],),
            self.event.is_set(),
            dtype=torch.bool,
            device=input_ids.device,
        )


class QwenModelBundle:
    """Load and share one Qwen model across RAG components."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )

        self.model.to(self.device)
        self.model.eval()

    def generate_tokens(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Generate text, optionally streaming pieces to on_token."""
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        if on_token is not None:
            return self._generate_streaming(
                inputs,
                max_new_tokens,
                on_token,
            )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][prompt_length:]

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    def _generate_streaming(
        self,
        inputs: dict,
        max_new_tokens: int,
        on_token: Callable[[str], None],
    ) -> str:
        """Run generation in a worker thread and relay text pieces."""
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        errors: list[BaseException] = []
        stop = Event()

        def run() -> None:
            try:
                # inference_mode is thread-local, so enter it here.
                with torch.inference_mode():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList(
                            [_StopWhenSet(stop)]
                        ),
                    )
            except BaseException as exc:
                errors.append(exc)
                # Unblock the consuming loop below.
                streamer.end()

        worker = Thread(target=run, daemon=True)
        worker.start()

        pieces: list[str] = []
        callback_error: BaseException | None = None
        for piece in streamer:
            if piece:
                pieces.append(piece)
                try:
                    on_token(piece)
                except BaseException as exc:
                    # The consumer gave up (for example the client
                    # disconnected): stop generating after this token.
                    callback_error = exc
                    stop.set()
                    break

        worker.join()

        if callback_error is not None:
            raise callback_error

        if errors:
            raise errors[0]

        return "".join(pieces).strip()
