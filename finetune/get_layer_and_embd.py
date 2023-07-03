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


model_file = open(sys.argv[1], "rb")
cleaner = file_cleaner(model_file)
cleaner_thread = threading.Thread(target=cleaner, daemon=True)
cleaner_thread.start()

w = torch.load(model_file, map_location="cpu")
gc.collect()

n_embd = w["emb.weight"].shape[1]
n_layer = 0
keys = list(w.keys())
for x in keys:
    layer_id = int(x.split(".")[1]) if ("blocks." in x) else 0
    n_layer = max(n_layer, layer_id + 1)

print(f"--n_layer {n_layer} --n_embd {n_embd}", end="")
