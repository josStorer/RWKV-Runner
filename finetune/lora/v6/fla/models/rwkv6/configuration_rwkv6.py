# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class RWKV6Config(PretrainedConfig):

    model_type = 'rwkv6'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = "chunk",
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        expand_k: int = 0.5,
        expand_v: int = 1,
        hidden_ratio: Optional[int] = 3.5,
        intermediate_size: Optional[int] = None,
        use_glu: Optional[bool] = False,
        num_hidden_layers: int = 24,
        num_heads: int = 4,
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        hidden_act: str = "sqrelu",
        max_position_embeddings: int = 2048,
        eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.use_glu = use_glu
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.attn_mode = attn_mode
        self.hidden_act = hidden_act
        self.eps = eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
