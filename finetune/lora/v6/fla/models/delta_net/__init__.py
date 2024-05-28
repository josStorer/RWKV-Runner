# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.delta_net.configuration_delta_net import \
    DeltaNetConfig
from fla.models.delta_net.modeling_delta_net import (
    DeltaNetForCausalLM, DeltaNetModel)

AutoConfig.register(DeltaNetConfig.model_type, DeltaNetConfig)
AutoModel.register(DeltaNetConfig, DeltaNetModel)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM)

__all__ = ['DeltaNetConfig', 'DeltaNetForCausalLM', 'DeltaNetModel']
