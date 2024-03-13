import torch
import sys
import time
import os
import threading
import gc


def file_cleaner(file):
    last_pos = 0

    def cleaner():
        nonlocal last_pos
        while True:
            time.sleep(0.1)
            pos = file.tell()
            if pos > last_pos:
                os.posix_fadvise(
                    file.fileno(), last_pos, pos - last_pos, os.POSIX_FADV_DONTNEED
                )
            last_pos = pos

    return cleaner


expected_max_version = float(sys.argv[2]) if len(sys.argv) > 2 else 100
model_file = open(sys.argv[1], "rb")
cleaner = file_cleaner(model_file)
cleaner_thread = threading.Thread(target=cleaner, daemon=True)
cleaner_thread.start()

w = torch.load(model_file, map_location="cpu")
gc.collect()

vocab_size = w["emb.weight"].shape[0]
n_embd = w["emb.weight"].shape[1]
n_layer = 0
keys = list(w.keys())
version = 4
for x in keys:
    layer_id = int(x.split(".")[1]) if ("blocks." in x) else 0
    n_layer = max(n_layer, layer_id + 1)

    if "ln_x" in x:
        version = max(5, version)
    if "gate.weight" in x:
        version = max(5.1, version)
    if int(version) == 5 and "att.time_decay" in x:
        if len(w[x].shape) > 1:
            if w[x].shape[1] > 1:
                version = max(5.2, version)
    if "time_maa" in x:
        version = max(6, version)

params = f"--vocab_size {vocab_size} --n_layer {n_layer} --n_embd {n_embd}"

if version <= expected_max_version:
    if version == 6:
        params += ' --my_testing "x060"'
    print(
        f"v{int(version)}/train.py {params}",
        end="",
    )
else:
    raise Exception(f"RWKV{version} is not supported")
