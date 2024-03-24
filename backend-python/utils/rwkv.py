from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type, Callable
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
from routes import state_cache
import global_var

os.environ["TORCH_EXTENSIONS_DIR"] = f"{pathlib.Path(__file__).parent.parent.resolve()}"


class RWKVType(Enum):
    NoneType = auto()
    Raven = auto()
    World = auto()
    Music = auto()


class AbstractRWKV(ABC):
    def __init__(self, model, pipeline):
        self.EOS_ID = 0

        self.name = "rwkv"
        self.version = 4
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
        self.penalty_decay = 0.996
        self.global_penalty = False

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
        import numpy as np

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
        import numpy as np

        quick_log(None, None, "Generation Prompt:\n" + prompt)
        cache = None
        delta_prompt = prompt
        try:
            cache = state_cache.longest_prefix_state(
                state_cache.LongestPrefixStateBody(prompt=prompt), None
            )
        except HTTPException:
            pass
        if cache is None or cache["prompt"] == "" or cache["state"] is None:
            self.model_state = None
            self.model_tokens = []
        else:
            delta_prompt = prompt[len(cache["prompt"]) :]
            self.model_state = cache["state"]
            self.model_tokens = cache["tokens"]
            logits = cache["logits"]

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

            if token == self.EOS_ID:
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
                        exit_flag = False
                        for s in stop:
                            if s in response:
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
                                exit_flag = True
                                response = response.split(s)[0]
                                yield response, "", prompt_token_len, completion_token_len
                                break
                        if exit_flag:
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

        self.AVOID_REPEAT_TOKENS = set()
        AVOID_REPEAT = "，：？！"
        for i in AVOID_REPEAT:
            dd = self.pipeline.encode(i)
            assert len(dd) == 1
            self.AVOID_REPEAT_TOKENS.add(dd[0])
        self.AVOID_PENALTY_TOKENS = set()
        AVOID_PENALTY = '\n,.:?!，。：？！"“”<>[]{}/\\|;；~`@#$%^&*()_+-=0123456789 '
        for i in AVOID_PENALTY:
            dd = self.pipeline.encode(i)
            if len(dd) == 1:
                self.AVOID_PENALTY_TOKENS.add(dd[0])

        self.__preload()

    def adjust_occurrence(self, occurrence: Dict, token: int):
        for xxx in occurrence:
            occurrence[xxx] *= self.penalty_decay
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        for n in occurrence:
            # if n not in self.AVOID_PENALTY_TOKENS:
            logits[n] -= (
                self.penalty_alpha_presence
                + occurrence[n] * self.penalty_alpha_frequency
            )

        # set global_penalty to False to get the same generated results as the official RWKV Gradio
        if self.global_penalty and i == 0:
            for token in self.model_tokens:
                token = int(token)
                if token not in self.AVOID_PENALTY_TOKENS:
                    self.adjust_occurrence(occurrence, token)

    # Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end
    def fix_tokens(self, tokens) -> List[int]:
        if self.rwkv_type == RWKVType.World:
            return tokens
        if len(tokens) > 0 and tokens[-1] == 535:
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


class MusicMidiRWKV(AbstractRWKV):
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


class MusicAbcRWKV(AbstractRWKV):
    def __init__(self, model, pipeline):
        super().__init__(model, pipeline)

        self.EOS_ID = 3

        self.max_tokens_per_generation = 500
        self.temperature = 1
        self.top_p = 0.8
        self.top_k = 8

        self.rwkv_type = RWKVType.Music

    def adjust_occurrence(self, occurrence: Dict, token: int):
        pass

    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        pass

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
        return delta


def get_tokenizer(tokenizer_len: int):
    tokenizer_dir = f"{pathlib.Path(__file__).parent.parent.resolve()}/rwkv_pip/"
    if tokenizer_len < 2176:
        return "abc_tokenizer"
    if tokenizer_len < 20096:
        return tokenizer_dir + "tokenizer-midipiano.json"
    if tokenizer_len < 50277:
        return tokenizer_dir + "tokenizer-midi.json"
    elif tokenizer_len < 65536:
        return tokenizer_dir + "20B_tokenizer.json"
    else:
        return "rwkv_vocab_v20230424"


def get_model_path(model_path: str) -> str:
    if os.path.isabs(model_path):
        return model_path

    working_dir: pathlib.Path = pathlib.Path(os.path.abspath(os.getcwd()))

    parent_paths: List[pathlib.Path] = [
        working_dir,  # [cwd](RWKV-Runner)/models/xxx
        working_dir.parent,  # [cwd](backend-python)/../models/xxx
        pathlib.Path(
            os.path.abspath(__file__)
        ).parent.parent,  # backend-python/models/xxx
        pathlib.Path(
            os.path.abspath(__file__)
        ).parent.parent.parent,  # RWKV-Runner/models/xxx
    ]

    child_paths: List[Callable[[pathlib.Path], pathlib.Path]] = [
        lambda p: p / model_path,
        lambda p: p / "build" / "bin" / model_path,  # for dev
    ]

    for parent_path in parent_paths:
        for child_path in child_paths:
            full_path: pathlib.Path = child_path(parent_path)

            if os.path.isfile(full_path):
                return str(full_path)

    return model_path


def RWKV(model: str, strategy: str, tokenizer: Union[str, None]) -> AbstractRWKV:
    model = get_model_path(model)

    rwkv_beta = global_var.get(global_var.Args).rwkv_beta
    rwkv_cpp = getattr(global_var.get(global_var.Args), "rwkv.cpp")
    webgpu = global_var.get(global_var.Args).webgpu

    if "midi" in model.lower() or "abc" in model.lower():
        os.environ["RWKV_RESCALE_LAYER"] = "999"

    # dynamic import to make RWKV_CUDA_ON work
    if rwkv_beta:
        print("Using rwkv-beta")
        from rwkv_pip.beta.model import (
            RWKV as Model,
        )
    elif rwkv_cpp:
        print("Using rwkv.cpp, strategy is ignored")
        from rwkv_pip.cpp.model import (
            RWKV as Model,
        )
    elif webgpu:
        print("Using webgpu")
        from rwkv_pip.webgpu.model import (
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
        "tokenizer-midi": MusicMidiRWKV,
        "tokenizer-midipiano": MusicMidiRWKV,
        "abc_tokenizer": MusicAbcRWKV,
    }
    tokenizer_name = os.path.splitext(os.path.basename(tokenizer))[0]
    global_var.set(
        global_var.Midi_Vocab_Config_Type,
        (
            global_var.MidiVocabConfig.Piano
            if tokenizer_name == "tokenizer-midipiano"
            else global_var.MidiVocabConfig.Default
        ),
    )
    rwkv: AbstractRWKV
    if tokenizer_name in rwkv_map:
        rwkv = rwkv_map[tokenizer_name](model, pipeline)
    else:
        tokenizer_name = tokenizer_name.lower()
        if "music" in tokenizer_name or "midi" in tokenizer_name:
            rwkv = MusicMidiRWKV(model, pipeline)
        elif "abc" in tokenizer_name:
            rwkv = MusicAbcRWKV(model, pipeline)
        else:
            rwkv = TextRWKV(model, pipeline)
    rwkv.name = filename
    rwkv.version = model.version

    return rwkv


class ModelConfigBody(BaseModel):
    max_tokens: int = Field(default=None, gt=0, le=102400)
    temperature: float = Field(default=None, ge=0, le=3)
    top_p: float = Field(default=None, ge=0, le=1)
    presence_penalty: float = Field(default=None, ge=-2, le=2)
    frequency_penalty: float = Field(default=None, ge=-2, le=2)
    penalty_decay: float = Field(default=None, ge=0.99, le=0.999)
    top_k: int = Field(default=None, ge=0, le=25)
    global_penalty: bool = Field(
        default=None,
        description="When generating a response, whether to include the submitted prompt as a penalty factor. By turning this off, you will get the same generated results as official RWKV Gradio. If you find duplicate results in the generated results, turning this on can help avoid generating duplicates.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "max_tokens": 1000,
                "temperature": 1,
                "top_p": 0.3,
                "presence_penalty": 0,
                "frequency_penalty": 1,
                "penalty_decay": 0.996,
                "global_penalty": False,
            }
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
    if body.penalty_decay is not None:
        model.penalty_decay = body.penalty_decay
    if body.top_k is not None:
        model.top_k = body.top_k
    if body.global_penalty is not None:
        model.global_penalty = body.global_penalty


def get_rwkv_config(model: AbstractRWKV) -> ModelConfigBody:
    return ModelConfigBody(
        max_tokens=model.max_tokens_per_generation,
        temperature=model.temperature,
        top_p=model.top_p,
        presence_penalty=model.penalty_alpha_presence,
        frequency_penalty=model.penalty_alpha_frequency,
        penalty_decay=model.penalty_decay,
        top_k=model.top_k,
        global_penalty=model.global_penalty,
    )
