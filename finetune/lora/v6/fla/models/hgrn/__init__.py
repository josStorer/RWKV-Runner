# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hgrn.configuration_hgrn import HGRNConfig
from fla.models.hgrn.modeling_hgrn import HGRNForCausalLM, HGRNModel

AutoConfig.register(HGRNConfig.model_type, HGRNConfig)
AutoModel.register(HGRNConfig, HGRNModel)
AutoModelForCausalLM.register(HGRNConfig, HGRNForCausalLM)


__all__ = ['HGRNConfig', 'HGRNForCausalLM', 'HGRNModel']
