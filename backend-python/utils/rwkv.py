from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
import numpy as np
from routes import state_cache
import global_var


END_OF_TEXT = 0
END_OF_LINE_DOUBLE = 535


os.environ["TORCH_EXTENSIONS_DIR"] = f"{pathlib.Path(__file__).parent.parent.resolve()}"


class RWKVType(Enum):
    NoneType = auto()
    Raven = auto()
    World = auto()
    Music = auto()


class AbstractRWKV(ABC):
    def __init__(self, model, pipeline):
        self.name = "rwkv"
        self.model = model
        self.pipeline = pipeline
        self.model_state = None
        self.model_tokens = []
        self.rwkv_type: RWKVType = RWKVType.NoneType
        self.tokenizer_len = len(model.w["emb.weight"])

        self.max_tokens_per_generation = 500
        self.temperature = 1
        self.top_p = 0.3
        self.top_k = 0
        self.penalty_alpha_presence = 0
        self.penalty_alpha_frequency = 1

    @abstractmethod
    def adjust_occurrence(self, occurrence: Dict, token: int):
        pass

    @abstractmethod
    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        pass

    # Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end
    @abstractmethod
    def fix_tokens(self, tokens) -> List[int]:
        pass

    @abstractmethod
    def run_rnn(
        self, _tokens: List[str], newline_adj: int = 0
    ) -> Tuple[List[float], int]:
        pass

    @abstractmethod
    def delta_postprocess(self, delta: str) -> str:
        pass

    def get_embedding(self, input: str, fast_mode: bool) -> Tuple[List[float], int]:
        if fast_mode:
            embedding, token_len = self.__fast_embedding(
                self.fix_tokens(self.pipeline.encode(input)), None
            )
        else:
            self.model_state = None
            self.model_tokens = []
            _, token_len = self.run_rnn(self.fix_tokens(self.pipeline.encode(input)))
            embedding = self.model_state[-11].tolist()
        embedding = (embedding / np.linalg.norm(embedding)).tolist()
        return embedding, token_len

    def __fast_embedding(self, tokens: List[str], state):
        import torch

        tokens = [int(x) for x in tokens]
        token_len = len(tokens)
        self = self.model

        with torch.no_grad():
            w = self.w
            args = self.args

            if state == None:
                state = [None] * args.n_layer * 5
                for i in range(
                    args.n_layer
                ):  # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
                    dd = self.strategy[i]
                    dev = dd.device
                    atype = dd.atype
                    state[i * 5 + 0] = torch.zeros(
                        args.n_embd, dtype=atype, requires_grad=False, device=dev
                    ).contiguous()
                    state[i * 5 + 1] = torch.zeros(
                        args.n_embd, dtype=torch.float, requires_grad=False, device=dev
                    ).contiguous()
                    state[i * 5 + 2] = torch.zeros(
                        args.n_embd, dtype=torch.float, requires_grad=False, device=dev
                    ).contiguous()
                    state[i * 5 + 3] = (
                        torch.zeros(
                            args.n_embd,
                            dtype=torch.float,
                            requires_grad=False,
                            device=dev,
                        ).contiguous()
                        - 1e30
                    )
                    state[i * 5 + 4] = torch.zeros(
                        args.n_embd, dtype=atype, requires_grad=False, device=dev
                    ).contiguous()

                    break

            seq_mode = len(tokens) > 1

            x = w["emb.weight"][tokens if seq_mode else tokens[0]]

            for i in range(args.n_layer):
                bbb = f"blocks.{i}."
                att = f"blocks.{i}.att."
                ffn = f"blocks.{i}.ffn."
                dd = self.strategy[i]
                dev = dd.device
                atype = dd.atype
                wtype = dd.wtype
                if seq_mode:
                    if "cuda" in str(dev) and os.environ["RWKV_CUDA_ON"] == "1":
                        ATT = (
                            self.cuda_att_seq
                            if wtype != torch.uint8
                            else self.cuda_att_seq_i8
                        )
                    else:
                        ATT = self.att_seq if wtype != torch.uint8 else self.att_seq_i8
                    FFN = self.ffn_seq if wtype != torch.uint8 else self.ffn_seq_i8
                else:
                    ATT = self.att_one if wtype != torch.uint8 else self.att_one_i8
                    FFN = self.ffn_one if wtype != torch.uint8 else self.ffn_one_i8

                x = x.to(dtype=atype, device=dev)

                kw = w[f"{att}key.weight"]
                vw = w[f"{att}value.weight"]
                rw = w[f"{att}receptance.weight"]
                ow = w[f"{att}output.weight"]
                if dd.stream:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)
                    ow = ow.to(device=dev, non_blocking=True)
                kmx = w[f"{att}key.weight_mx"] if wtype == torch.uint8 else x
                krx = w[f"{att}key.weight_rx"] if wtype == torch.uint8 else x
                kmy = w[f"{att}key.weight_my"] if wtype == torch.uint8 else x
                kry = w[f"{att}key.weight_ry"] if wtype == torch.uint8 else x
                vmx = w[f"{att}value.weight_mx"] if wtype == torch.uint8 else x
                vrx = w[f"{att}value.weight_rx"] if wtype == torch.uint8 else x
                vmy = w[f"{att}value.weight_my"] if wtype == torch.uint8 else x
                vry = w[f"{att}value.weight_ry"] if wtype == torch.uint8 else x
                rmx = w[f"{att}receptance.weight_mx"] if wtype == torch.uint8 else x
                rrx = w[f"{att}receptance.weight_rx"] if wtype == torch.uint8 else x
                rmy = w[f"{att}receptance.weight_my"] if wtype == torch.uint8 else x
                rry = w[f"{att}receptance.weight_ry"] if wtype == torch.uint8 else x
                omx = w[f"{att}output.weight_mx"] if wtype == torch.uint8 else x
                orx = w[f"{att}output.weight_rx"] if wtype == torch.uint8 else x
                omy = w[f"{att}output.weight_my"] if wtype == torch.uint8 else x
                ory = w[f"{att}output.weight_ry"] if wtype == torch.uint8 else x
                (
                    x,
                    state[i * 5 + 0],
                    state[i * 5 + 1],
                    state[i * 5 + 2],
                    state[i * 5 + 3],
                ) = ATT(
                    x,
                    state[i * 5 + 0],
                    state[i * 5 + 1],
                    state[i * 5 + 2],
                    state[i * 5 + 3],
                    w[f"{bbb}ln1.weight"],
                    w[f"{bbb}ln1.bias"],
                    w[f"{att}time_mix_k"],
                    w[f"{att}time_mix_v"],
                    w[f"{att}time_mix_r"],
                    w[f"{att}time_decay"],
                    w[f"{att}time_first"],
                    kw,
                    vw,
                    rw,
                    ow,
                    kmx,
                    krx,
                    kmy,
                    kry,
                    vmx,
                    vrx,
                    vmy,
                    vry,
                    rmx,
                    rrx,
                    rmy,
                    rry,
                    omx,
                    orx,
                    omy,
                    ory,
                )

                return state[0].tolist(), token_len

    def generate(
        self, prompt: str, stop: Union[str, List[str], None] = None
    ) -> Iterable[Tuple[str, str, int, int]]:
        quick_log(None, None, "Generation Prompt:\n" + prompt)
        cache = None
        delta_prompt = prompt
        try:
            cache = state_cache.longest_prefix_state(
                state_cache.LongestPrefixStateBody(prompt=prompt), None
            )
        except HTTPException:
            pass
        if cache is None or cache["prompt"] == "":
            self.model_state = None
            self.model_tokens = []
        else:
            delta_prompt = prompt[len(cache["prompt"]) :]
            self.model_state = copy.deepcopy(cache["state"])
            self.model_tokens = copy.deepcopy(cache["tokens"])
            logits = copy.deepcopy(cache["logits"])

        prompt_token_len = 0
        if delta_prompt != "":
            logits, prompt_token_len = self.run_rnn(
                self.fix_tokens(self.pipeline.encode(delta_prompt))
            )
            try:
                state_cache.add_state(
                    state_cache.AddStateBody(
                        prompt=prompt,
                        tokens=self.model_tokens,
                        state=self.model_state,
                        logits=logits,
                    )
                )
            except HTTPException:
                pass

        begin = len(self.model_tokens)
        out_last = begin

        occurrence: Dict = {}

        completion_token_len = 0
        response = ""
        for i in range(self.max_tokens_per_generation):
            self.adjust_forward_logits(logits, occurrence, i)

            token = self.pipeline.sample_logits(
                logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k
            )

            if token == END_OF_TEXT:
                yield response, "", prompt_token_len, completion_token_len
                break

            self.adjust_occurrence(occurrence, token)

            logits, _ = self.run_rnn([token])
            completion_token_len = completion_token_len + 1
            delta: str = self.delta_postprocess(
                self.pipeline.decode(self.model_tokens[out_last:])
            )
            if "\ufffd" not in delta:  # avoid utf-8 display issues
                response += delta
                if stop is not None:
                    if type(stop) == str:
                        if stop in response:
                            try:
                                state_cache.add_state(
                                    state_cache.AddStateBody(
                                        prompt=prompt + response,
                                        tokens=self.model_tokens,
                                        state=self.model_state,
                                        logits=logits,
                                    )
                                )
                            except HTTPException:
                                pass
                            response = response.split(stop)[0]
                            yield response, "", prompt_token_len, completion_token_len
                            break
                    elif type(stop) == list:
                        stop_exist_regex = "|".join(stop)
                        matched = re.search(stop_exist_regex, response)
                        if matched:
                            try:
                                state_cache.add_state(
                                    state_cache.AddStateBody(
                                        prompt=prompt + response,
                                        tokens=self.model_tokens,
                                        state=self.model_state,
                                        logits=logits,
                                    )
                                )
                            except HTTPException:
                                pass
                            response = response.split(matched.group())[0]
                            yield response, "", prompt_token_len, completion_token_len
                            break
                out_last = begin + i + 1
                if i == self.max_tokens_per_generation - 1:
                    try:
                        state_cache.add_state(
                            state_cache.AddStateBody(
                                prompt=prompt + response,
                                tokens=self.model_tokens,
                                state=self.model_state,
                                logits=logits,
                            )
                        )
                    except HTTPException:
                        pass
                yield response, delta, prompt_token_len, completion_token_len


class TextRWKV(AbstractRWKV):
    def __init__(self, model, pipeline) -> None:
        super().__init__(model, pipeline)

        self.CHUNK_LEN = 256

        self.max_tokens_per_generation = 500
        self.temperature = 1
        self.top_p = 0.3
        self.top_k = 0
        self.penalty_alpha_presence = 0
        self.penalty_alpha_frequency = 1

        self.interface = ":"
        if self.tokenizer_len < 65536:
            self.rwkv_type = RWKVType.Raven
            self.user = "Bob"
            self.bot = "Alice"
            self.END_OF_LINE = 187
        else:
            self.rwkv_type = RWKVType.World
            self.user = "User"
            self.bot = "Assistant"
            self.END_OF_LINE = 11

        self.AVOID_REPEAT_TOKENS = []
        AVOID_REPEAT = "，：？！"
        for i in AVOID_REPEAT:
            dd = self.pipeline.encode(i)
            assert len(dd) == 1
            self.AVOID_REPEAT_TOKENS += dd

        self.__preload()

    def adjust_occurrence(self, occurrence: Dict, token: int):
        for xxx in occurrence:
            occurrence[xxx] *= 0.996
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        for n in occurrence:
            logits[n] -= (
                self.penalty_alpha_presence
                + occurrence[n] * self.penalty_alpha_frequency
            )

        if i == 0:
            for token in self.model_tokens:
                token = int(token)
                for xxx in occurrence:
                    occurrence[xxx] *= 0.996
                if token not in occurrence:
                    occurrence[token] = 1
                else:
                    occurrence[token] += 1

    # Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end
    def fix_tokens(self, tokens) -> List[int]:
        if self.rwkv_type == RWKVType.World:
            return tokens
        if len(tokens) > 0 and tokens[-1] == END_OF_LINE_DOUBLE:
            tokens = tokens[:-1] + [self.END_OF_LINE, self.END_OF_LINE]
        return tokens

    def run_rnn(
        self, _tokens: List[str], newline_adj: int = 0
    ) -> Tuple[List[float], int]:
        tokens = [int(x) for x in _tokens]
        token_len = len(tokens)
        self.model_tokens += tokens

        while len(tokens) > 0:
            out, self.model_state = self.model.forward(
                tokens[: self.CHUNK_LEN], self.model_state
            )
            tokens = tokens[self.CHUNK_LEN :]

        out[self.END_OF_LINE] += newline_adj  # adjust \n probability

        if self.model_tokens[-1] in self.AVOID_REPEAT_TOKENS:
            out[self.model_tokens[-1]] = -999999999
        return out, token_len

    def delta_postprocess(self, delta: str) -> str:
        return delta

    def __preload(self):
        interface = self.interface
        user = self.user
        bot = self.bot
        preset_system = (
            f"""
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
{bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. \
{bot} likes to tell {user} a lot about herself and her opinions. \
{bot} usually gives {user} kind, helpful and informative advices.\n
"""
            if self.rwkv_type == RWKVType.Raven
            else (
                f"{user}{interface} hi\n\n{bot}{interface} Hi. "
                + "I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
            )
        )
        logits, _ = self.run_rnn(self.fix_tokens(self.pipeline.encode(preset_system)))
        try:
            state_cache.add_state(
                state_cache.AddStateBody(
                    prompt=preset_system,
                    tokens=self.model_tokens,
                    state=self.model_state,
                    logits=logits,
                )
            )
        except HTTPException:
            pass


class MusicRWKV(AbstractRWKV):
    def __init__(self, model, pipeline):
        super().__init__(model, pipeline)

        self.max_tokens_per_generation = 500
        self.temperature = 1
        self.top_p = 0.8
        self.top_k = 8

        self.rwkv_type = RWKVType.Music

    def adjust_occurrence(self, occurrence: Dict, token: int):
        for n in occurrence:
            occurrence[n] *= 0.997  #### decay repetition penalty
        if token >= 128 or token == 127:
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        else:
            occurrence[token] = 0.3 + (occurrence[token] if token in occurrence else 0)

    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        for n in occurrence:
            logits[n] -= 0 + occurrence[n] * 0.5

        logits[0] += (i - 2000) / 500  # try not to be too short or too long
        logits[127] -= 1  # avoid "t125"

    def fix_tokens(self, tokens) -> List[int]:
        return tokens

    def run_rnn(
        self, _tokens: List[str], newline_adj: int = 0
    ) -> Tuple[List[float], int]:
        tokens = [int(x) for x in _tokens]
        token_len = len(tokens)
        self.model_tokens += tokens
        out, self.model_state = self.model.forward(tokens, self.model_state)
        return out, token_len

    def delta_postprocess(self, delta: str) -> str:
        return " " + delta


def get_tokenizer(tokenizer_len: int):
    tokenizer_dir = f"{pathlib.Path(__file__).parent.parent.resolve()}/rwkv_pip/"
    if tokenizer_len < 50277:
        return tokenizer_dir + "tokenizer-midi.json"
    elif tokenizer_len < 65536:
        return tokenizer_dir + "20B_tokenizer.json"
    else:
        return "rwkv_vocab_v20230424"


def RWKV(model: str, strategy: str, tokenizer: Union[str, None]) -> AbstractRWKV:
    rwkv_beta = global_var.get(global_var.Args).rwkv_beta

    # dynamic import to make RWKV_CUDA_ON work
    if rwkv_beta:
        from rwkv_pip.beta.model import (
            RWKV as Model,
        )
    else:
        from rwkv_pip.model import (
            RWKV as Model,
        )
    from rwkv_pip.utils import PIPELINE

    filename, _ = os.path.splitext(os.path.basename(model))
    model = Model(model, strategy)
    if not tokenizer:
        tokenizer = get_tokenizer(len(model.w["emb.weight"]))
    pipeline = PIPELINE(model, tokenizer)

    rwkv_map: dict[str, Type[AbstractRWKV]] = {
        "20B_tokenizer": TextRWKV,
        "rwkv_vocab_v20230424": TextRWKV,
        "tokenizer-midi": MusicRWKV,
    }
    tokenizer_name = os.path.splitext(os.path.basename(tokenizer))[0]
    rwkv: AbstractRWKV
    if tokenizer_name in rwkv_map:
        rwkv = rwkv_map[tokenizer_name](model, pipeline)
    else:
        rwkv = TextRWKV(model, pipeline)
    rwkv.name = filename

    return rwkv


class ModelConfigBody(BaseModel):
    max_tokens: int = Field(default=None, gt=0, le=102400)
    temperature: float = Field(default=None, ge=0, le=2)
    top_p: float = Field(default=None, ge=0, le=1)
    presence_penalty: float = Field(default=None, ge=-2, le=2)
    frequency_penalty: float = Field(default=None, ge=-2, le=2)

    class Config:
        json_schema_extra = {
            "example": {
                "max_tokens": 1000,
                "temperature": 1.2,
                "top_p": 0.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
            }
        }


def set_rwkv_config(model: AbstractRWKV, body: ModelConfigBody):
    if body.max_tokens is not None:
        model.max_tokens_per_generation = body.max_tokens
    if body.temperature is not None:
        if body.temperature < 0.1:
            model.temperature = 0.1
        else:
            model.temperature = body.temperature
    if body.top_p is not None:
        model.top_p = body.top_p
    if body.presence_penalty is not None:
        model.penalty_alpha_presence = body.presence_penalty
    if body.frequency_penalty is not None:
        model.penalty_alpha_frequency = body.frequency_penalty


def get_rwkv_config(model: AbstractRWKV) -> ModelConfigBody:
    return ModelConfigBody(
        max_tokens=model.max_tokens_per_generation,
        temperature=model.temperature,
        top_p=model.top_p,
        presence_penalty=model.penalty_alpha_presence,
        frequency_penalty=model.penalty_alpha_frequency,
    )
