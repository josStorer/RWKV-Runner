from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch
import bitsandbytes as bnb
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base_model", default="", type=str)
parser.add_argument("--state_checkpoint", default="", type=str)
parser.add_argument("--output", default="", type=str)
# parser.add_argument("--quant", default="none", type=str)
parser.add_argument("--device", default="cuda", type=str)
# parser.add_argument("--lora_alpha", default=16, type=int)
args = parser.parse_args()
device= args.device
base_model = args.base_model
state= args.state_checkpoint
output= args.output


with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_state: Dict[str, torch.Tensor] = torch.load(state, map_location='cpu')

    for k in w_state.keys():
        print(k)
        w[k] = w_state[k]
    # merge LoRA weights
    for k in w.keys():
        print(k)
    
    torch.save(w, output)