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


class ChatCompletionBody(ModelConfigBody):
    messages: List[Message]
    model: str = "rwkv"
    stream: bool = False
    stop: str = None


completion_lock = Lock()


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionBody, request: Request):
    model: RWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    question = body.messages[-1]
    if question.role == "user":
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no question found")

    interface = model.interface
    user = model.user
    bot = model.bot

    completion_text = (
        f"""
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
{bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. \
{bot} likes to tell {user} a lot about herself and her opinions. \
{bot} usually gives {user} kind, helpful and informative advices.\n
"""
        if user == "Bob"
        else ""
    )
    for message in body.messages:
        if message.role == "system":
            completion_text = (
                f"The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. "
                if user == "Bob"
                else ""
                + message.content.replace("\\n", "\n")
                .replace("\r\n", "\n")
                .replace("\n\n", "\n")
                .replace("\n", " ")
                .strip()
                .replace("You are", f"{bot} is")
                .replace("you are", f"{bot} is")
                .replace("You're", f"{bot} is")
                .replace("you're", f"{bot} is")
                .replace("You", f"{bot}")
                .replace("you", f"{bot}")
                .replace("Your", f"{bot}'s")
                .replace("your", f"{bot}'s")
                .replace("你", f"{bot}")
                + "\n\n"
            )
        elif message.role == "user":
            completion_text += (
                f"{user}{interface} "
                + message.content.replace("\\n", "\n")
                .replace("\r\n", "\n")
                .replace("\n\n", "\n")
                .strip()
                + "\n\n"
            )
        elif message.role == "assistant":
            completion_text += (
                f"{bot}{interface} "
                + message.content.replace("\\n", "\n")
                .replace("\r\n", "\n")
                .replace("\n\n", "\n")
                .strip()
                + "\n\n"
            )
    completion_text += f"{bot}{interface}"

    async def eval_rwkv():
        while completion_lock.locked():
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.1)
        else:
            completion_lock.acquire()
            set_rwkv_config(model, global_var.get(global_var.Model_Config))
            set_rwkv_config(model, body)
            if body.stream:
                for response, delta in model.generate(
                    completion_text,
                    stop=f"\n\n{user}" if body.stop is None else body.stop,
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
                # torch_gc()
                completion_lock.release()
                if await request.is_disconnected():
                    return
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
                for response, delta in model.generate(
                    completion_text,
                    stop=f"\n\n{user}" if body.stop is None else body.stop,
                ):
                    if await request.is_disconnected():
                        break
                # torch_gc()
                completion_lock.release()
                if await request.is_disconnected():
                    return
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

    if body.stream:
        return EventSourceResponse(eval_rwkv())
    else:
        return await eval_rwkv().__anext__()


class CompletionBody(ModelConfigBody):
    prompt: str
    model: str = "rwkv"
    stream: bool = False
    stop: str = None


@router.post("/v1/completions")
@router.post("/completions")
async def completions(body: CompletionBody, request: Request):
    model: RWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.prompt is None or body.prompt == "":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "prompt not found")

    async def eval_rwkv():
        while completion_lock.locked():
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.1)
        else:
            completion_lock.acquire()
            set_rwkv_config(model, global_var.get(global_var.Model_Config))
            set_rwkv_config(model, body)
            if body.stream:
                for response, delta in model.generate(body.prompt, stop=body.stop):
                    if await request.is_disconnected():
                        break
                    yield json.dumps(
                        {
                            "response": response,
                            "model": "rwkv",
                            "choices": [
                                {
                                    "text": delta,
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                # torch_gc()
                completion_lock.release()
                if await request.is_disconnected():
                    return
                yield json.dumps(
                    {
                        "response": response,
                        "model": "rwkv",
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                yield "[DONE]"
            else:
                response = None
                for response, delta in model.generate(body.prompt, stop=body.stop):
                    if await request.is_disconnected():
                        break
                # torch_gc()
                completion_lock.release()
                if await request.is_disconnected():
                    return
                yield {
                    "response": response,
                    "model": "rwkv",
                    "choices": [
                        {
                            "text": response,
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                }

    if body.stream:
        return EventSourceResponse(eval_rwkv())
    else:
        return await eval_rwkv().__anext__()
