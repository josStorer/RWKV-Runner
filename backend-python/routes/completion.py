import asyncio
import json
from threading import Lock
from typing import List

from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from utils.rwkv import *
import global_var

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class CompletionBody(ModelConfigBody):
    messages: List[Message]
    model: str = "rwkv"
    stream: bool = False


completion_lock = Lock()


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def completions(body: CompletionBody, request: Request):
    model: RWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    question = body.messages[-1]
    if question.role == "user":
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no question found")

    completion_text = ""
    for message in body.messages:
        if message.role == "user":
            completion_text += "Bob: " + message.content + "\n\n"
        elif message.role == "assistant":
            completion_text += "Alice: " + message.content + "\n\n"
    completion_text += "Alice:"

    async def eval_rwkv():
        while completion_lock.locked():
            await asyncio.sleep(0.1)
        else:
            with completion_lock:
                set_rwkv_config(model, global_var.get(global_var.Model_Config))
                set_rwkv_config(model, body)
                if body.stream:
                    for response, delta in rwkv_generate(
                        model, completion_text, stop="\n\nBob"
                    ):
                        if await request.is_disconnected():
                            break
                        yield json.dumps(
                            {
                                "response": response,
                                "model": "rwkv",
                                "choices": [
                                    {
                                        "delta": {"content": delta},
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    yield json.dumps(
                        {
                            "response": response,
                            "model": "rwkv",
                            "choices": [
                                {
                                    "delta": {},
                                    "index": 0,
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    )
                    yield "[DONE]"
                else:
                    response = None
                    for response, delta in rwkv_generate(
                        model, completion_text, stop="\n\nBob"
                    ):
                        pass
                    yield {
                        "response": response,
                        "model": "rwkv",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": response,
                                },
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                # torch_gc()

    if body.stream:
        return EventSourceResponse(eval_rwkv())
    else:
        return await eval_rwkv().__anext__()
