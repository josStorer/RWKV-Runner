import json
import os
import sys
import copy
import torch
from safetensors.torch import load_file, save_file

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Path to input pth model")
parser.add_argument(
    "--output",
    type=str,
    default="./converted.st",
    help="Path to output safetensors model",
)
args = parser.parse_args()


def rename_key(rename, name):
    for k, v in rename.items():
        if k in name:
            name = name.replace(k, v)
    return name


def convert_file(pt_filename: str, sf_filename: str, transpose_names=[], rename={}):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]

    loaded = {k: v.clone().half() for k, v in loaded.items()}
    # for k, v in loaded.items():
    #     print(f'{k}\t{v.shape}\t{v.dtype}')

    # For tensors to be contiguous
    for k, v in loaded.items():
        for transpose_name in transpose_names:
            if transpose_name in k:
                loaded[k] = v.transpose(0, 1)
    loaded = {rename_key(rename, k).lower(): v.contiguous() for k, v in loaded.items()}

    for k, v in loaded.items():
        print(f"{k}\t{v.shape}\t{v.dtype}")

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


if __name__ == "__main__":
    try:
        convert_file(
            args.input,
            args.output,
            ["lora_A"],
            {"time_faaaa": "time_first", "lora_A": "lora.0", "lora_B": "lora.1"},
        )
        print(f"Saved to {args.output}")
    except Exception as e:
        with open("error.txt", "w") as f:
            f.write(str(e))
