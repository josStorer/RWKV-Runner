import json
import pathlib
import sys
from typing import List
import os
import sysconfig

from fastapi import FastAPI, Request, status, HTTPException
from langchain.llms import RWKV
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rwkv_helper import rwkv_generate


def set_torch():
    torch_path = os.path.join(sysconfig.get_paths()["purelib"], "torch\\lib")
    paths = os.environ.get("PATH", "")
    if os.path.exists(torch_path):
        print(f"torch found: {torch_path}")
        if torch_path in paths:
            print("torch already set")
        else:
            print("run:")
            os.environ['PATH'] = paths + os.pathsep + torch_path + os.pathsep
            print(f'set Path={paths + os.pathsep + torch_path + os.pathsep}')
    else:
        print("torch not found")


def torch_gc():
    import torch

    if torch.cuda.is_available():
        with torch.cuda.device(0):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event('startup')
def init():
    global model

    set_torch()

    model = RWKV(
        model=sys.argv[2],
        strategy=sys.argv[1],
        tokens_path=f"{pathlib.Path(__file__).parent.resolve()}/20B_tokenizer.json"
    )

    if os.environ.get("ngrok_token") is not None:
        ngrok_connect()


def ngrok_connect():
    from pyngrok import ngrok, conf
    conf.set_default(conf.PyngrokConfig(ngrok_path="./ngrok"))
    ngrok.set_auth_token(os.environ["ngrok_token"])
    http_tunnel = ngrok.connect(8000)
    print(http_tunnel.public_url)


class Message(BaseModel):
    role: str
    content: str


class Body(BaseModel):
    messages: List[Message]
    model: str
    stream: bool
    max_tokens: int


@app.get("/")
def read_root():
    return {"Hello": "World!"}

@app.post("update-config")
def updateConfig(body: Body):
    pass


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def completions(body: Body, request: Request):
    global model

    question = body.messages[-1]
    if question.role == 'user':
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    completion_text = ""
    for message in body.messages:
        if message.role == 'user':
            completion_text += "Bob: " + message.content + "\n\n"
        elif message.role == 'assistant':
            completion_text += "Alice: " + message.content + "\n\n"
    completion_text += "Alice:"

    async def eval_rwkv():
        if body.stream:
            for response, delta in rwkv_generate(model, completion_text):
                if await request.is_disconnected():
                    break
                yield json.dumps({"response": response, "choices": [{"delta": {"content": delta}}], "model": "rwkv"})
            yield "[DONE]"
        else:
            response = None
            for response, delta in rwkv_generate(model, completion_text):
                pass
            yield json.dumps({"response": response, "model": "rwkv"})
        # torch_gc()

    return EventSourceResponse(eval_rwkv())


if __name__ == "__main__":
    uvicorn.run("main:app", reload=False, app_dir="backend-python")
