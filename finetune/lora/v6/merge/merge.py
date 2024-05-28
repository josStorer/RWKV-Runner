from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch
import bitsandbytes as bnb
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--type", default="pissa", type=str)
parser.add_argument("--base_model", default="", type=str)
parser.add_argument("--lora_init", default="none", type=str)
parser.add_argument("--lora_checkpoint", default="", type=str)
parser.add_argument("--output", default="", type=str)
parser.add_argument("--quant", default="none", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--lora_alpha", default=16, type=int)
args = parser.parse_args()
device= args.device
base_model = args.base_model
init_lora= args.lora_init
lora= args.lora_checkpoint
output= args.output
quant= args.quant
lora_alpha = args.lora_alpha

with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')

    if args.type=='pissa':
        w_init_lora: Dict[str, torch.Tensor] = torch.load(init_lora, map_location='cpu')
    for k in w_lora.keys():
        w[k] = w_lora[k]
    output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
    # merge LoRA weights
    keys = list(w.keys())
    for k in keys:
        if k.endswith('.weight'):
            prefix = k[:-len('.weight')]
            lora_A = prefix + '.lora_A'
            lora_B = prefix + '.lora_B'
            init_lora_A = prefix + '.init_lora_A'
            init_lora_B = prefix + '.init_lora_B'
            if lora_A in keys:
                assert lora_B in keys
                print(f'merging {lora_A} and {lora_B} into {k}')
                assert w[lora_B].shape[1] == w[lora_A].shape[0]
                lora_r = w[lora_B].shape[1]
                w[k] = w[k].to(device=device)
                w[lora_A] = w[lora_A].to(device=device)
                w[lora_B] = w[lora_B].to(device=device)
                
                if args.type=='pissa':
                    w_init_lora[init_lora_A] = w_init_lora[init_lora_A].to(device=device)
                    w_init_lora[init_lora_B] = w_init_lora[init_lora_B].to(device=device)
                    if quant=='4bit':
                        qw,qs = bnb.functional.quantize_4bit(w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A])
                        w[k] = (bnb.functional.dequantize_4bit(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant == 'nf4':
                        qw,qs = bnb.functional.quantize_nf4(w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A])
                        w[k] = (bnb.functional.dequantize_nf4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant == 'fp4':
                        qw,qs = bnb.functional.quantize_fp4(w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A])
                        w[k] = (bnb.functional.dequantize_fp4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    else:
                        w[k] = (w[k]- w_init_lora[init_lora_B] @ w_init_lora[init_lora_A]).to(dtype=torch.bfloat16)
                    w[k] +=  w[lora_B] @ w[lora_A]
                else:
                    if quant=='4bit':
                        qw,qs = bnb.functional.quantize_4bit(w[k])
                        w[k] = (bnb.functional.dequantize_4bit(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant=='nf4':
                        qw,qs = bnb.functional.quantize_nf4(w[k])
                        w[k] = (bnb.functional.dequantize_nf4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    elif quant=='fp4':
                        qw,qs = bnb.functional.quantize_fp4(w[k])
                        w[k] = (bnb.functional.dequantize_fp4(qw,quant_state=qs)).to(dtype=torch.bfloat16)
                    w[k] += w[lora_B] @ w[lora_A] * (lora_alpha / lora_r)
                output_w[k] = w[k].to(device='cpu', copy=True)
                del w[k]
                del w[lora_A]
                del w[lora_B]
                continue

        if 'lora' not in k:
            print(f'retaining {k}')
            output_w[k] = w[k].clone()
            del w[k]
    torch.save(output_w, output)