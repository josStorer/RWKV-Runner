from typing import Dict
from langchain.llms import RWKV


def rwkv_generate(model: RWKV, prompt: str):
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
            yield response, delta
            out_last = begin + i + 1
            if i >= model.max_tokens_per_generation - 100:
                break
