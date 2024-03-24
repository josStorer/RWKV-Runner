from typing import Any, List, Union
from . import rwkv_cpp_model
from . import rwkv_cpp_shared_library


class RWKV:
    def __init__(self, model_path: str, strategy=None):
        self.library = rwkv_cpp_shared_library.load_rwkv_shared_library()
        self.model = rwkv_cpp_model.RWKVModel(self.library, model_path)
        self.w = {}  # fake weight
        self.w["emb.weight"] = [0] * self.model.n_vocab
        self.version = (
            self.model.arch_version_major + self.model.arch_version_minor / 10
        )

    def forward(self, tokens: List[int], state: Union[Any, None] = None):
        return self.model.eval_sequence_in_chunks(tokens, state, use_numpy=True)
