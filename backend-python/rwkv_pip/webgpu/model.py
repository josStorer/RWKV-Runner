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
        layer = (
            int(s.lstrip("layer"))
            for s in strategy.split()
            for s in s.split(",")
            if s.startswith("layer")
        )

        chunk_size = (
            int(s.lstrip("chunk"))
            for s in strategy.split()
            for s in s.split(",")
            if s.startswith("chunk")
        )
        self.token_chunk_size = next(chunk_size, 32)

        args = {
            "path": model_path,
            "quant": next(layer, 31) if "i8" in strategy else 0,
            "quant_nf4": next(layer, 26) if "i4" in strategy else 0,
        }
        self.model = wrp.Model(**args)
        self.info = self.model.info()
        self.w = {}  # fake weight
        self.w["emb.weight"] = [0] * self.info.num_vocab
        self.version = str(self.info.version).lower()
        self.version = float(self.version.lower().replace("v", ""))

    def forward(self, tokens: List[int], state: Union[Any, None] = None):
        if state is None:
            self.model.clear_state()
        elif type(state).__name__ == "State_Cpu":
            self.model.load_state(state)
        logits = self.model.run(tokens, self.token_chunk_size)
        ret_state = "State_Gpu"
        return logits, ret_state
