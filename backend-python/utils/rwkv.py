import os
import pathlib
from typing import Dict
from langchain.llms import RWKV
from pydantic import BaseModel


class ModelConfigBody(BaseModel):
    max_tokens: int = None
    temperature: float = None
    top_p: float = None
    presence_penalty: float = None
    frequency_penalty: float = None


def set_rwkv_config(model: RWKV, body: ModelConfigBody):
    if body.max_tokens:
        model.max_tokens_per_generation = body.max_tokens
    if body.temperature:
        model.temperature = body.temperature
    if body.top_p:
        model.top_p = body.top_p
    if body.presence_penalty:
        model.penalty_alpha_presence = body.presence_penalty
    if body.frequency_penalty:
        model.penalty_alpha_frequency = body.frequency_penalty


def get_rwkv_config(model: RWKV) -> ModelConfigBody:
    return ModelConfigBody(
        max_tokens=model.max_tokens_per_generation,
        temperature=model.temperature,
        top_p=model.top_p,
        presence_penalty=model.penalty_alpha_presence,
        frequency_penalty=model.penalty_alpha_frequency,
    )


os.environ["TORCH_EXTENSIONS_DIR"] = f"{pathlib.Path(__file__).parent.parent.resolve()}"


def rwkv_generate(model: RWKV, prompt: str, stop: str = None):
    model.model_state = None
    model.model_tokens = []
    logits = model.run_rnn(model.tokenizer.encode(prompt).ids)
    begin = len(model.model_tokens)
    out_last = begin

    occurrence: Dict = {}

    response = ""
    for i in range(model.max_tokens_per_generation):
        for n in occurrence:
            logits[n] -= (
                model.penalty_alpha_presence
                + occurrence[n] * model.penalty_alpha_frequency
            )
        token = model.pipeline.sample_logits(
            logits, temperature=model.temperature, top_p=model.top_p
        )

        END_OF_TEXT = 0
        if token == END_OF_TEXT:
            break
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        logits = model.run_rnn([token])
        delta: str = model.tokenizer.decode(model.model_tokens[out_last:])
        if "\ufffd" not in delta:  # avoid utf-8 display issues
            response += delta
            if stop is not None:
                if stop in response:
                    response = response.split(stop)[0]
                    yield response, ""
                    break
            yield response, delta
            out_last = begin + i + 1
            if i >= model.max_tokens_per_generation - 100:
                break
