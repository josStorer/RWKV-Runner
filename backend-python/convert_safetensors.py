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


def convert_file(pt_filename: str, sf_filename: str, rename={}, transpose_names=[]):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]

    kk = list(loaded.keys())
    version = 4
    for x in kk:
        if "ln_x" in x:
            version = max(5, version)
        if "gate.weight" in x:
            version = max(5.1, version)
        if int(version) == 5 and "att.time_decay" in x:
            if len(loaded[x].shape) > 1:
                if loaded[x].shape[1] > 1:
                    version = max(5.2, version)
        if "time_maa" in x:
            version = max(6, version)

    if version == 5.1 and "midi" in pt_filename.lower():
        import numpy as np

        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        kk = list(loaded.keys())
        _, n_emb = loaded["emb.weight"].shape
        for k in kk:
            if "time_decay" in k or "time_faaaa" in k:
                # print(k, mm[k].shape)
                loaded[k] = (
                    loaded[k].unsqueeze(1).repeat(1, n_emb // loaded[k].shape[0])
                )

    loaded = {k: v.clone().half() for k, v in loaded.items()}
    # for k, v in loaded.items():
    #     print(f'{k}\t{v.shape}\t{v.dtype}')

    loaded = {rename_key(rename, k).lower(): v.contiguous() for k, v in loaded.items()}
    # For tensors to be contiguous
    for k, v in loaded.items():
        for transpose_name in transpose_names:
            if transpose_name in k:
                loaded[k] = v.transpose(0, 1)

    loaded = {k: v.clone().half().contiguous() for k, v in loaded.items()}

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
            rename={
                "time_faaaa": "time_first",
                "time_maa": "time_mix",
                "lora_A": "lora.0",
                "lora_B": "lora.1",
            },
            transpose_names=[
                "time_mix_w1",
                "time_mix_w2",
                "time_decay_w1",
                "time_decay_w2",
            ],
        )
        print(f"Saved to {args.output}")
    except Exception as e:
        print(e)
        with open("error.txt", "w") as f:
            f.write(str(e))
