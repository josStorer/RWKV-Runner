import json
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


class CompletionBody(BaseModel):
    messages: List[Message]
    model: str
    stream: bool
    max_tokens: int


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def completions(body: CompletionBody, request: Request):
    model = global_var.get(global_var.Model)
    if (model is None):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    question = body.messages[-1]
    if question.role == 'user':
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no question found")

    completion_text = ""
    for message in body.messages:
        if message.role == 'user':
            completion_text += "Bob: " + message.content + "\n\n"
        elif message.role == 'assistant':
            completion_text += "Alice: " + message.content + "\n\n"
    completion_text += "Alice:"

    async def eval_rwkv():
        if body.stream:
            for response, delta in rwkv_generate(model, completion_text, stop="Bob:"):
                if await request.is_disconnected():
                    break
                yield json.dumps({"response": response, "choices": [{"delta": {"content": delta}}], "model": "rwkv"})
            yield "[DONE]"
        else:
            response = None
            for response, delta in rwkv_generate(model, completion_text, stop="Bob:"):
                pass
            yield json.dumps({"response": response, "model": "rwkv"})
        # torch_gc()

    return EventSourceResponse(eval_rwkv())
