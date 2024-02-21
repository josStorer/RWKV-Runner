from typing import Any, List, Union

try:
    import web_rwkv_py as wrp
except ModuleNotFoundError:
    try:
        from . import web_rwkv_py as wrp
    except ImportError:
        raise ModuleNotFoundError(
            "web_rwkv_py not found, install it from https://github.com/cryscan/web-rwkv-py"
        )


class RWKV:
    def __init__(self, model_path: str, strategy: str = None):
        self.info = wrp.peek_info(model_path)
        self.w = {}  # fake weight
        self.w["emb.weight"] = [0] * self.info.num_vocab
        self.version = str(self.info.version).lower()
        self.wrp = getattr(wrp, self.version)

        args = {
            "file": model_path,
            "turbo": True,
            "quant": 31 if "i8" in strategy else 0,
            "quant_nf4": 26 if "i4" in strategy else 0,
            "token_chunk_size": 32,
            "lora": None,
        }
        self.model = self.wrp.Model(**args)

    def forward(self, tokens: List[int], state: Union[Any, None] = None):
        if type(state).__name__ == "BackedState":  # memory state
            gpu_state = self.wrp.ModelState(self.model, 1)
            gpu_state.load(state)
        else:
            gpu_state = state
        return self.wrp.run_one(self.model, tokens, gpu_state)
