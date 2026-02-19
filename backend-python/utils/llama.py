from abc import ABC, abstractmethod
import os
from typing import Iterable, Iterator, List, Literal, Tuple, Union

from fastapi.encoders import jsonable_encoder
from utils.log import quick_log
from utils.rwkv import ModelConfigBody, get_model_path, AbstractRWKV


class AbstractLlama(ABC):
    def __init__(self, model):
        self.name = "llama"
        self.model_path = ""
        self.version = 100
        self.model = model

        self.max_tokens_per_generation = 500
        self.temperature = 1.0
        self.top_p = 0.3
        self.top_k = 40
        self.penalty_alpha_presence = 0.0
        self.penalty_alpha_frequency = 0.0
        self.stateless = False

    @abstractmethod
    def delta_postprocess(self, delta: str) -> str:
        pass

    def generate(
        self,
        body: ModelConfigBody,
        prompt: str,
        stop: Union[str, List[str], None] = None,
        stop_token_ids: Union[List[int], None] = None,
    ) -> Iterable[Tuple[Literal["text", "tool"], str, str, int, int]]:
        quick_log(None, None, "Generation Prompt:\n" + prompt)
        completion_token_len = 0
        response = ""


        from routes.completion import ChatCompletionBody

        if not is_rwkv_model(self) and isinstance(body, ChatCompletionBody):
            from llama_cpp import CreateChatCompletionStreamResponse

            stream_chat: Iterator[CreateChatCompletionStreamResponse] = (
                self.model.create_chat_completion(
                    messages=body.messages,
                    tools=jsonable_encoder(body.tools) if body.tools else None,
                    tool_choice=body.tool_choice,
                    max_tokens=self.max_tokens_per_generation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stream=True,
                    stop=stop,
                    frequency_penalty=self.penalty_alpha_frequency,
                    presence_penalty=self.penalty_alpha_presence,
                )
            )

            for chunk in stream_chat:
                if not chunk.get("choices"):
                    continue
                completion_token_len = completion_token_len + 1
                delta = chunk["choices"][0].get("delta", {})
                content = self.delta_postprocess(delta.get("content", ""))

                if content:
                    response += content
                    yield "text", response, content, 0, completion_token_len
                # for tool in delta.get("tool_calls", []) or []:
                #     yield "tool", response, json.dumps(
                #         tool["function"]
                #     ), 0, completion_token_len
        else:
            from llama_cpp import CreateCompletionStreamResponse

            if self.stateless:
                self.clear_rwkv_state()

            stream: Iterator[CreateCompletionStreamResponse] = (
                self.model.create_completion(
                    prompt=prompt,
                    max_tokens=self.max_tokens_per_generation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stream=True,
                    stop=stop,
                    frequency_penalty=self.penalty_alpha_frequency,
                    presence_penalty=self.penalty_alpha_presence,
                )
            )

            for chunk in stream:
                if not chunk.get("choices"):
                    continue
                completion_token_len = completion_token_len + 1
                delta = self.delta_postprocess(chunk["choices"][0].get("text", ""))
                response += delta

                yield "text", response, delta, 0, completion_token_len

    def clear_rwkv_state(self):
        """Properly clear RWKV recurrent state and library cache"""
        if not is_rwkv_model(self):
            raise ValueError("clear_rwkv_state is only applicable for RWKV models.")

        # 1. Use the official llama-cpp-python reset method.
        self.model.reset()
        
        # 2. Ensure both the wrapper and the library object are zeroed
        self.model.n_tokens = 0

    def get_rwkv_state(self) -> Tuple[bytes, int, int]:
        """Extracts the RWKV state, size, and token count using llama_cpp C API."""
        if not is_rwkv_model(self):
            raise ValueError("get_rwkv_state is only applicable for RWKV models.")

        if self.stateless:
            raise ValueError("Model is configured as stateless; state extraction is not applicable.")

        import llama_cpp
        import ctypes
        
        ctx_ptr = self.model._ctx.ctx
        state_size = llama_cpp.llama_get_state_size(ctx_ptr)
        
        state_data = (ctypes.c_uint8 * state_size)()
        llama_cpp.llama_copy_state_data(ctx_ptr, state_data)
        
        return bytes(state_data), state_size, self.model.n_tokens

    def set_rwkv_state(self, state_bytes: bytes, tokens: int):
        """Injects the RWKV state state and updates the token count."""
        if not is_rwkv_model(self):
            raise ValueError("set_rwkv_state is only applicable for RWKV models.")

        if self.stateless:
            raise ValueError("Model is configured as stateless; state injection is not applicable.")

        import llama_cpp
        import ctypes
        
        ctx_ptr = self.model._ctx.ctx
        
        # Clear existing state/memory before loading new one
        self.clear_rwkv_state()
        
        # Inject the state back into the C context
        state_data = (ctypes.c_uint8 * len(state_bytes)).from_buffer_copy(state_bytes)
        llama_cpp.llama_set_state_data(ctx_ptr, state_data)
        
        # Prime the context
        self.model.n_tokens = tokens


class TextLlama(AbstractLlama):
    def __init__(self, model) -> None:
        super().__init__(model)

        self.max_tokens_per_generation = 500
        self.temperature = 1.0
        self.top_p = 0.3
        self.top_k = 40
        self.penalty_alpha_presence = 0.0
        self.penalty_alpha_frequency = 1.0

        self.interface = ":"
        self.user = "User"
        self.bot = "Assistant"

        self.__preload()

    def delta_postprocess(self, delta: str) -> str:
        return delta

    def __preload(self):
        pass    


def Llama(model_path: str, strategy: str) -> AbstractLlama:
    model_path = get_model_path(model_path)

    from llama_cpp import Llama as LlamaCpp

    filename, _ = os.path.splitext(os.path.basename(model_path))
    n_ctx = 8192
    try:
        n_ctx = int(strategy.split(" ")[1])
    except:
        pass

    # Check if this is an RWKV model
    is_rwkv = "rwkv" in filename.lower()
    
    if is_rwkv:
        # RWKV models need reset=False to maintain sequential RNN state
        class RWKVLlama(LlamaCpp):
            """Llama wrapper that forces reset=False for RWKV's sequential state"""
            def generate(self, tokens, reset=False, **kwargs):
                # Always use reset=False for RWKV to avoid state position mismatches
                return super().generate(tokens, reset=False, **kwargs)
        
        model = RWKVLlama(
            model_path, 
            n_gpu_layers=-1 if "cpu" not in strategy else 0, 
            n_ctx=n_ctx
        )
    else:
        model = LlamaCpp(
            model_path, 
            n_gpu_layers=-1 if "cpu" not in strategy else 0, 
            n_ctx=n_ctx
        )

    llama: AbstractLlama
    llama = TextLlama(model)
    llama.name = filename
    llama.model_path = model_path

    return llama


def set_llama_config(model: AbstractLlama, body: ModelConfigBody):
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
    if body.top_k is not None:
        model.top_k = body.top_k


def get_llama_config(model: AbstractLlama) -> ModelConfigBody:
    return ModelConfigBody(
        max_tokens=model.max_tokens_per_generation,
        temperature=model.temperature,
        top_p=model.top_p,
        presence_penalty=model.penalty_alpha_presence,
        frequency_penalty=model.penalty_alpha_frequency,
        top_k=model.top_k,
    )


# you can rename gguf file to rwkv to use builtin rwkv prompt template
def is_rwkv_model(model: Union[AbstractRWKV, AbstractLlama]) -> bool:
    return isinstance(model, AbstractRWKV) or "rwkv" in model.name.lower()
