import time

start_time = time.time()

import argparse
from typing import Union, Sequence


def get_args(args: Union[Sequence[str], None] = None):
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="server arguments")
    group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to run the server on (default: 8000)",
    )
    group.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="host to run the server on (default: 127.0.0.1)",
    )
    group = parser.add_argument_group(title="mode arguments")
    group.add_argument(
        "--webui",
        action="store_true",
        help="whether to enable WebUI (default: False)",
    )
    group.add_argument(
        "--rwkv-beta",
        action="store_true",
        help="whether to use rwkv-beta (default: False)",
    )
    group.add_argument(
        "--rwkv.cpp",
        action="store_true",
        help="whether to use rwkv.cpp (default: False)",
    )
    group.add_argument(
        "--webgpu",
        action="store_true",
        help="whether to use webgpu (default: False)",
    )
    args = parser.parse_args(args)

    return args


if __name__ == "__main__":
    args = get_args()


import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import psutil
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils.rwkv import *
from utils.torch import *
from utils.ngrok import *
from utils.log import log_middleware
from routes import completion, config, state_cache, midi, misc, file_process
import global_var


@asynccontextmanager
async def lifespan(app: FastAPI):
    init()
    yield


app = FastAPI(lifespan=lifespan, dependencies=[Depends(log_middleware)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(completion.router)
app.include_router(config.router)
app.include_router(midi.router)
app.include_router(file_process.router)
app.include_router(misc.router)
app.include_router(state_cache.router)


@app.post("/exit", tags=["Root"])
def exit():
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

    parent_pid = os.getpid()
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


try:
    if (
        "RWKV_RUNNER_PARAMS" in os.environ
        and "--webui" in os.environ["RWKV_RUNNER_PARAMS"].split(" ")
    ) or args.webui:
        from webui_server import webui_server

        app.mount("/", webui_server)
except NameError:
    pass


@app.get("/", tags=["Root"])
def read_root():
    return {"Hello": "World!"}


def init():
    global_var.init()
    cmd_params = os.environ["RWKV_RUNNER_PARAMS"]
    global_var.set(
        global_var.Args, get_args(cmd_params.split(" ") if cmd_params else None)
    )

    state_cache.init()

    set_torch()

    if os.environ.get("ngrok_token") is not None:
        ngrok_connect()


if __name__ == "__main__":
    os.environ["RWKV_RUNNER_PARAMS"] = " ".join(sys.argv[1:])
    print("--- %s seconds ---" % (time.time() - start_time))
    uvicorn.run("main:app", port=args.port, host=args.host, workers=1)
