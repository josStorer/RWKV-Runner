# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.linear_attn.configuration_linear_attn import \
    LinearAttentionConfig
from fla.models.linear_attn.modeling_linear_attn import (
    LinearAttentionForCausalLM, LinearAttentionModel)

AutoConfig.register(LinearAttentionConfig.model_type, LinearAttentionConfig)
AutoModel.register(LinearAttentionConfig, LinearAttentionModel)
AutoModelForCausalLM.register(LinearAttentionConfig, LinearAttentionForCausalLM)

__all__ = ['LinearAttentionConfig', 'LinearAttentionForCausalLM', 'LinearAttentionModel']
