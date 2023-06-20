import asyncio
import json
from threading import Lock
from typing import List
import base64

from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import numpy as np
import tiktoken
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


class CompletionBody(ModelConfigBody):
    prompt: str
    model: str = "rwkv"
    stream: bool = False
    stop: str = None

    class Config:
        schema_extra = {
            "example": {
                "prompt": "The following is an epic science fiction masterpiece that is immortalized, "
                + "with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n",
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


completion_lock = Lock()

requests_num = 0


async def eval_rwkv(
    model: RWKV,
    request: Request,
    body: ModelConfigBody,
    prompt: str,
    stream: bool,
    stop: str,
    chat_mode: bool,
):
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

        response, prompt_tokens, completion_tokens = "", 0, 0
        for response, delta, prompt_tokens, completion_tokens in model.generate(
            prompt,
            stop=stop,
        ):
            if await request.is_disconnected():
                break
            if stream:
                yield json.dumps(
                    {
                        "object": "chat.completion.chunk"
                        if chat_mode
                        else "text_completion",
                        "response": response,
                        "model": model.name,
                        "choices": [
                            {
                                "delta": {"content": delta},
                                "index": 0,
                                "finish_reason": None,
                            }
                            if chat_mode
                            else {
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
        if stream:
            yield json.dumps(
                {
                    "object": "chat.completion.chunk"
                    if chat_mode
                    else "text_completion",
                    "response": response,
                    "model": model.name,
                    "choices": [
                        {
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop",
                        }
                        if chat_mode
                        else {
                            "text": "",
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            yield "[DONE]"
        else:
            yield {
                "object": "chat.completion" if chat_mode else "text_completion",
                "response": response,
                "model": model.name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response,
                        },
                        "index": 0,
                        "finish_reason": "stop",
                    }
                    if chat_mode
                    else {
                        "text": response,
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
            }


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
        else f"{user}{interface} hi\n\n{bot}{interface} Hi. "
        + "I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
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

    stop = f"\n\n{user}" if body.stop is None else body.stop
    if body.stream:
        return EventSourceResponse(
            eval_rwkv(model, request, body, completion_text, body.stream, stop, True)
        )
    else:
        try:
            return await eval_rwkv(
                model, request, body, completion_text, body.stream, stop, True
            ).__anext__()
        except StopAsyncIteration:
            return None


@router.post("/v1/completions")
@router.post("/completions")
async def completions(body: CompletionBody, request: Request):
    model: RWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.prompt is None or body.prompt == "":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "prompt not found")

    if body.stream:
        return EventSourceResponse(
            eval_rwkv(model, request, body, body.prompt, body.stream, body.stop, False)
        )
    else:
        try:
            return await eval_rwkv(
                model, request, body, body.prompt, body.stream, body.stop, False
            ).__anext__()
        except StopAsyncIteration:
            return None


class EmbeddingsBody(BaseModel):
    input: str | List[str] | List[List[int]]
    model: str = "rwkv"
    encoding_format: str = None
    fast_mode: bool = False

    class Config:
        schema_extra = {
            "example": {
                "input": "a big apple",
                "model": "rwkv",
                "encoding_format": None,
                "fast_mode": False,
            }
        }


def embedding_base64(embedding: List[float]) -> str:
    return base64.b64encode(np.array(embedding).astype(np.float32)).decode("utf-8")


@router.post("/v1/embeddings")
@router.post("/embeddings")
@router.post("/v1/engines/text-embedding-ada-002/embeddings")
@router.post("/engines/text-embedding-ada-002/embeddings")
async def embeddings(body: EmbeddingsBody, request: Request):
    model: RWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.input is None or body.input == "" or body.input == [] or body.input == [[]]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "input not found")

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

        base64_format = False
        if body.encoding_format == "base64":
            base64_format = True

        embeddings = []
        prompt_tokens = 0
        if type(body.input) == list:
            if type(body.input[0]) == list:
                encoding = tiktoken.model.encoding_for_model("text-embedding-ada-002")
                for i in range(len(body.input)):
                    if await request.is_disconnected():
                        break
                    input = encoding.decode(body.input[i])
                    embedding, token_len = model.get_embedding(input, body.fast_mode)
                    prompt_tokens = prompt_tokens + token_len
                    if base64_format:
                        embedding = embedding_base64(embedding)
                    embeddings.append(embedding)
            else:
                for i in range(len(body.input)):
                    if await request.is_disconnected():
                        break
                    embedding, token_len = model.get_embedding(
                        body.input[i], body.fast_mode
                    )
                    prompt_tokens = prompt_tokens + token_len
                    if base64_format:
                        embedding = embedding_base64(embedding)
                    embeddings.append(embedding)
        else:
            embedding, prompt_tokens = model.get_embedding(body.input, body.fast_mode)
            if base64_format:
                embedding = embedding_base64(embedding)
            embeddings.append(embedding)

        requests_num = requests_num - 1
        completion_lock.release()
        if await request.is_disconnected():
            print(f"{request.client} Stop Waiting")
            quick_log(
                request,
                None,
                "Stop Waiting. RequestsNum: " + str(requests_num),
            )
            return
        quick_log(
            request,
            None,
            "Finished. RequestsNum: " + str(requests_num),
        )

        ret_data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": embedding,
            }
            for i, embedding in enumerate(embeddings)
        ]

        return {
            "object": "list",
            "data": ret_data,
            "model": model.name,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
