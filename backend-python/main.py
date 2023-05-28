import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils.rwkv import *
from utils.torch import *
from utils.ngrok import *
from routes import completion, config, state_cache
import global_var

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(completion.router)
app.include_router(config.router)
app.include_router(state_cache.router)


@app.on_event("startup")
def init():
    global_var.init()
    state_cache.init()

    set_torch()

    if os.environ.get("ngrok_token") is not None:
        ngrok_connect()


@app.get("/")
def read_root():
    return {"Hello": "World!", "pid": os.getpid()}


@app.post("/exit")
def exit():
    parent_pid = os.getpid()
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


def debug():
    model = RWKV(
        model="../models/RWKV-4-Raven-7B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230430-ctx8192.pth",
        strategy="cuda fp16",
        tokens_path="20B_tokenizer.json",
    )
    d = model.tokenizer.decode([])
    print(d)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000 if len(sys.argv) < 2 else int(sys.argv[1]),
        host="127.0.0.1" if len(sys.argv) < 3 else sys.argv[2],
    )
    # debug()
