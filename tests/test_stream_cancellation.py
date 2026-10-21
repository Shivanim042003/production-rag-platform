import asyncio
import queue
import time
from unittest.mock import patch

import pytest
import torch

from rag import main, qwen_model
from rag.qwen_model import QwenModelBundle


def test_closing_the_stream_cancels_generation():
    outcome = {}

    class FakeService:
        def query(self, question, on_token):
            on_token("first")
            for _ in range(300):
                time.sleep(0.01)
                try:
                    on_token("more")
                except main.GenerationCancelled:
                    outcome["cancelled"] = True
                    raise
            outcome["cancelled"] = False
            return {"answer": "x", "grounded": True, "sources": []}

    async def scenario():
        with patch.object(main, "get_rag_service", return_value=FakeService()):
            stream = main.rag_event_stream("question")
            await anext(stream)  # status
            await anext(stream)  # first token
            await stream.aclose()
            await asyncio.sleep(0.5)

    asyncio.run(scenario())

    assert outcome["cancelled"] is True


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}


class FakeStreamer:
    def __init__(self, tokenizer, **kwargs):
        self.items = queue.Queue()

    def put_text(self, text):
        self.items.put(text)

    def end(self):
        self.items.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.items.get()
        if item is None:
            raise StopIteration
        return item


class FakeModel:
    def __init__(self):
        self.steps = 0

    def generate(self, input_ids, max_new_tokens, do_sample, streamer,
                 stopping_criteria=None, **kwargs):
        for index in range(max_new_tokens):
            self.steps += 1
            streamer.put_text(f"t{index} ")
            time.sleep(0.005)
            if stopping_criteria is not None and bool(
                stopping_criteria(input_ids, None).any()
            ):
                break
        streamer.end()


def test_callback_error_stops_generation_early():
    bundle = object.__new__(QwenModelBundle)
    bundle.tokenizer = FakeTokenizer()
    bundle.model = FakeModel()
    bundle.device = "cpu"

    def on_token(piece):
        raise RuntimeError("client gone")

    with patch.object(qwen_model, "TextIteratorStreamer", FakeStreamer):
        with pytest.raises(RuntimeError, match="client gone"):
            bundle.generate_tokens(
                [{"role": "user", "content": "hi"}],
                max_new_tokens=1000,
                on_token=on_token,
            )

    assert bundle.model.steps < 200