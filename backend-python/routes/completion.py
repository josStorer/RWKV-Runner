import asyncio
import json
from threading import Lock
from typing import List

from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from utils.rwkv import *
from utils.log import quick_log
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

    class Config:
        schema_extra = {
            "example": {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "rwkv",
                "stream": False,
                "stop": None,
                "max_tokens": 1000,
                "temperature": 1.2,
                "top_p": 0.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
            }
        }


completion_lock = Lock()

requests_num = 0


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionBody, request: Request):
    model: RWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    question = body.messages[-1]
    if question.role == "user":
        question = question.content
    elif question.role == "system":
        question = body.messages[-2]
        if question.role == "user":
            question = question.content
        else:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "no question found")
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
        else f"{user}{interface} hi\n\n{bot}{interface} Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
    )
    for message in body.messages:
        if message.role == "system":
            completion_text = (
                f"The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. "
                if user == "Bob"
                else f"{user}{interface} hi\n\n{bot}{interface} Hi. "
                + message.content.replace("\\n", "\n")
                .replace("\r\n", "\n")
                .replace("\n\n", "\n")
                .replace("\n", " ")
                .strip()
                .replace("You are", f"{bot} is" if user == "Bob" else "I am")
                .replace("you are", f"{bot} is" if user == "Bob" else "I am")
                .replace("You're", f"{bot} is" if user == "Bob" else "I'm")
                .replace("you're", f"{bot} is" if user == "Bob" else "I'm")
                .replace("You", f"{bot}" if user == "Bob" else "I")
                .replace("you", f"{bot}" if user == "Bob" else "I")
                .replace("Your", f"{bot}'s" if user == "Bob" else "My")
                .replace("your", f"{bot}'s" if user == "Bob" else "my")
                .replace("你", f"{bot}" if user == "Bob" else "我")
                + "\n\n"
            )
            break
    for message in body.messages:
        if message.role == "user":
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
        global requests_num
        requests_num = requests_num + 1
        quick_log(request, None, "Start Waiting. RequestsNum: " + str(requests_num))
        while completion_lock.locked():
            if await request.is_disconnected():
                requests_num = requests_num - 1
                print(f"{request.client} Stop Waiting (Lock)")
                quick_log(
                    request,
                    None,
                    "Stop Waiting (Lock). RequestsNum: " + str(requests_num),
                )
                return
            await asyncio.sleep(0.1)
        else:
            completion_lock.acquire()
            if await request.is_disconnected():
                completion_lock.release()
                requests_num = requests_num - 1
                print(f"{request.client} Stop Waiting (Lock)")
                quick_log(
                    request,
                    None,
                    "Stop Waiting (Lock). RequestsNum: " + str(requests_num),
                )
                return
            set_rwkv_config(model, global_var.get(global_var.Model_Config))
            set_rwkv_config(model, body)
            if body.stream:
                response = ""
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
                requests_num = requests_num - 1
                completion_lock.release()
                if await request.is_disconnected():
                    print(f"{request.client} Stop Waiting")
                    quick_log(
                        request,
                        body,
                        response + "\nStop Waiting. RequestsNum: " + str(requests_num),
                    )
                    return
                quick_log(
                    request,
                    body,
                    response + "\nFinished. RequestsNum: " + str(requests_num),
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
                response = ""
                for response, delta in model.generate(
                    completion_text,
                    stop=f"\n\n{user}" if body.stop is None else body.stop,
                ):
                    if await request.is_disconnected():
                        break
                # torch_gc()
                requests_num = requests_num - 1
                completion_lock.release()
                if await request.is_disconnected():
                    print(f"{request.client} Stop Waiting")
                    quick_log(
                        request,
                        body,
                        response + "\nStop Waiting. RequestsNum: " + str(requests_num),
                    )
                    return
                quick_log(
                    request,
                    body,
                    response + "\nFinished. RequestsNum: " + str(requests_num),
                )
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
        try:
            return await eval_rwkv().__anext__()
        except StopAsyncIteration:
            return None


class CompletionBody(ModelConfigBody):
    prompt: str
    model: str = "rwkv"
    stream: bool = False
    stop: str = None

    class Config:
        schema_extra = {
            "example": {
                "prompt": "The following is an epic science fiction masterpiece that is immortalized, with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n",
                "model": "rwkv",
                "stream": False,
                "stop": None,
                "max_tokens": 100,
                "temperature": 1.2,
                "top_p": 0.5,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4,
            }
        }


@router.post("/v1/completions")
@router.post("/completions")
async def completions(body: CompletionBody, request: Request):
    model: RWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.prompt is None or body.prompt == "":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "prompt not found")

    async def eval_rwkv():
        global requests_num
        requests_num = requests_num + 1
        quick_log(request, None, "Start Waiting. RequestsNum: " + str(requests_num))
        while completion_lock.locked():
            if await request.is_disconnected():
                requests_num = requests_num - 1
                print(f"{request.client} Stop Waiting (Lock)")
                quick_log(
                    request,
                    None,
                    "Stop Waiting (Lock). RequestsNum: " + str(requests_num),
                )
                return
            await asyncio.sleep(0.1)
        else:
            completion_lock.acquire()
            if await request.is_disconnected():
                completion_lock.release()
                requests_num = requests_num - 1
                print(f"{request.client} Stop Waiting (Lock)")
                quick_log(
                    request,
                    None,
                    "Stop Waiting (Lock). RequestsNum: " + str(requests_num),
                )
                return
            set_rwkv_config(model, global_var.get(global_var.Model_Config))
            set_rwkv_config(model, body)
            if body.stream:
                response = ""
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
                requests_num = requests_num - 1
                completion_lock.release()
                if await request.is_disconnected():
                    print(f"{request.client} Stop Waiting")
                    quick_log(
                        request,
                        body,
                        response + "\nStop Waiting. RequestsNum: " + str(requests_num),
                    )
                    return
                quick_log(
                    request,
                    body,
                    response + "\nFinished. RequestsNum: " + str(requests_num),
                )
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
                response = ""
                for response, delta in model.generate(body.prompt, stop=body.stop):
                    if await request.is_disconnected():
                        break
                # torch_gc()
                requests_num = requests_num - 1
                completion_lock.release()
                if await request.is_disconnected():
                    print(f"{request.client} Stop Waiting")
                    quick_log(
                        request,
                        body,
                        response + "\nStop Waiting. RequestsNum: " + str(requests_num),
                    )
                    return
                quick_log(
                    request,
                    body,
                    response + "\nFinished. RequestsNum: " + str(requests_num),
                )
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
        try:
            return await eval_rwkv().__anext__()
        except StopAsyncIteration:
            return None
