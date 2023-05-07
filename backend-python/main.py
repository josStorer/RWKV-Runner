import os
import psutil

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils.rwkv import *
from utils.torch import *
from utils.ngrok import *
from routes import completion, config
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


@app.on_event('startup')
def init():
    global_var.init()

    set_torch()

    if os.environ.get("ngrok_token") is not None:
        ngrok_connect()


@app.get("/")
def read_root():
    return {"Hello": "World!"}


@app.post("/exit")
def exit():
    parent_pid = os.getpid()
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000)
