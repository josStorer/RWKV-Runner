import asyncio
import json
from threading import Lock
from typing import List, Union, Literal
from enum import Enum
import base64
import time, ast

from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import tiktoken

from routes.schema import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionNamedToolChoiceParam,
)
from utils.rwkv import *
from utils.log import quick_log
import global_var

router = APIRouter()


class Role(Enum):
    User = "user"
    Assistant = "assistant"
    System = "system"
    Tool = "tool"


default_stop = [
    "\n\nUser",
    "\n\nQuestion",
    "\n\nQ",
    "\n\nHuman",
    "\n\nBob",
    "\n\nAssistant",
    "\n\nAnswer",
    "\n\nA",
    "\n\nBot",
    "\n\nAlice",
]


class ChatCompletionBody(ModelConfigBody):
    messages: Union[List[ChatCompletionMessageParam], None]
    model: Union[str, None] = "rwkv"
    stream: bool = False
    stop: Union[str, List[str], None] = default_stop
    tools: Union[List[ChatCompletionToolParam], None] = None
    tool_choice: Union[
        Literal["none", "auto", "required"], ChatCompletionNamedToolChoiceParam
    ] = "auto"
    user_name: Union[str, None] = Field(
        None, description="Internal user name", min_length=1
    )
    assistant_name: Union[str, None] = Field(
        None, description="Internal assistant name", min_length=1
    )
    system_name: Union[str, None] = Field(
        None, description="Internal system name", min_length=1
    )
    presystem: bool = Field(
        False, description="Whether to insert default system prompt at the beginning"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": Role.User.value, "content": "hello", "raw": False}
                ],
                "model": "rwkv",
                "stream": False,
                "stop": None,
                "user_name": None,
                "assistant_name": None,
                "system_name": None,
                "presystem": True,
                "max_tokens": 1000,
                "temperature": 1,
                "top_p": 0.3,
                "presence_penalty": 0,
                "frequency_penalty": 1,
            }
        }
    }


class CompletionBody(ModelConfigBody):
    prompt: Union[str, List[str], None]
    model: Union[str, None] = "rwkv"
    stream: bool = False
    stop: Union[str, List[str], None] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "The following is an epic science fiction masterpiece that is immortalized, "
                + "with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n",
                "model": "rwkv",
                "stream": False,
                "stop": None,
                "max_tokens": 100,
                "temperature": 1,
                "top_p": 0.3,
                "presence_penalty": 0,
                "frequency_penalty": 1,
            }
        }
    }


completion_lock = Lock()

requests_num = 0


async def eval_rwkv(
    model: AbstractRWKV,
    request: Request,
    body: ModelConfigBody | ChatCompletionBody,
    prompt: str,
    stream: bool,
    stop: Union[str, List[str], None],
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
        with completion_lock:
            if await request.is_disconnected():
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
            print(get_rwkv_config(model))

            response, prompt_tokens, completion_tokens = "", 0, 0
            completion_start_time = None
            for response, delta, prompt_tokens, completion_tokens in model.generate(
                prompt,
                stop=stop,
            ):
                if not completion_start_time:
                    completion_start_time = time.time()
                if await request.is_disconnected():
                    break
                if stream:
                    yield json.dumps(
                        {
                            "object": (
                                "chat.completion.chunk"
                                if chat_mode
                                else "text_completion"
                            ),
                            # "response": response,
                            "model": model.name,
                            "choices": [
                                (
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
                                )
                            ],
                        }
                    )
            # torch_gc()
            requests_num = requests_num - 1
            completion_end_time = time.time()
            completion_interval = completion_end_time - completion_start_time
            tps = 0
            if completion_interval > 0:
                tps = completion_tokens / completion_interval
            print(f"Generation TPS: {tps:.2f}")

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
                        "object": (
                            "chat.completion.chunk" if chat_mode else "text_completion"
                        ),
                        # "response": response,
                        "model": model.name,
                        "choices": [
                            (
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
                            )
                        ],
                    }
                )
                yield "[DONE]"
            else: # !stream
                if isinstance(body, ChatCompletionBody):
                    yield{
                        "id": "",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model.name,
                        "choices": [
                            (
                                {
                                    "index": 0,
                                    "message": {
                                        "role": Role.Assistant.value,
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "",
                                                "type": "function",
                                                "function": ast.literal_eval(response)
                                            }
                                        ],
                                    },
                                    "logprobs": None,
                                    "finish_reason": "tool_calls",
                                } if isinstance(body.tools, List)
                                else {                  # body is None
                                    "text": response,
                                    "index": 0,
                                    "finish_reason": "stop",
                                }
                            )
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                else: # !isinstance(body, ChatCompletionBody)
                    yield {
                        "object": "text_completion",
                        # "response": response,
                        "model": model.name,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                        "choices": [
                            {
                                "message": {
                                    "role": Role.Assistant.value,
                                    "content": response,
                                },
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        ],
                    }


def chat_template_old(
    model: TextRWKV, body: ChatCompletionBody, interface: str, user: str, bot: str
):
    is_raven = model.rwkv_type == RWKVType.Raven

    completion_text: str = ""
    basic_system: Union[str, None] = None
    if body.presystem:
        if body.messages[0].role == Role.System:
            basic_system = body.messages[0].content

        if basic_system is None:
            completion_text = (
                f"""
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
{bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. \
{bot} likes to tell {user} a lot about herself and her opinions. \
{bot} usually gives {user} kind, helpful and informative advices.\n
"""
                if is_raven
                else (
                    f"{user}{interface} hi\n\n{bot}{interface} Hi. "
                    + "I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
                )
            )
        else:
            if not body.messages[0].raw:
                basic_system = (
                    basic_system.replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .replace("\n\n", "\n")
                    .replace("\n", " ")
                    .strip()
                )
            completion_text = (
                (
                    f"The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. "
                    if is_raven
                    else f"{user}{interface} hi\n\n{bot}{interface} Hi. "
                )
                + basic_system.replace("You are", f"{bot} is" if is_raven else "I am")
                .replace("you are", f"{bot} is" if is_raven else "I am")
                .replace("You're", f"{bot} is" if is_raven else "I'm")
                .replace("you're", f"{bot} is" if is_raven else "I'm")
                .replace("You", f"{bot}" if is_raven else "I")
                .replace("you", f"{bot}" if is_raven else "I")
                .replace("Your", f"{bot}'s" if is_raven else "My")
                .replace("your", f"{bot}'s" if is_raven else "my")
                .replace("你", f"{bot}" if is_raven else "我")
                + "\n\n"
            )

    for message in body.messages[(0 if basic_system is None else 1) :]:
        append_message: str = ""
        if message.role == Role.User:
            append_message = f"{user}{interface} " + message.content
        elif message.role == Role.Assistant:
            append_message = f"{bot}{interface} " + message.content
        elif message.role == Role.System:
            append_message = message.content
        if not message.raw:
            append_message = (
                append_message.replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\n\n", "\n")
                .strip()
            )
        completion_text += append_message + "\n\n"
    completion_text += f"{bot}{interface}"

    return completion_text


def chat_template(
    model: TextRWKV, body: ChatCompletionBody, interface: str, user: str, bot: str
):
    completion_text: str = ""
    if body.presystem:
        completion_text = (
            f"{user}{interface} hi\n\n{bot}{interface} Hi. "
            + "I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
        )

    system = "System" if body.system_name is None else body.system_name
    tool = "Obersavtion"
    for message in body.messages:
        append_message: str = ""
        if message.role == Role.User.value:
            append_message = f"{user}{interface} " + message.content
        elif message.role == Role.Assistant.value:
            append_message = f"{bot}{interface} " + message.content
        elif message.role == Role.System.value:
            append_message = f"{system}{interface} " + message.content
        elif message.role == Role.Tool.value:
            append_message = f"{tool}{interface} " + message.content
        completion_text += append_message + "\n\n"
    completion_text += f"{bot}{interface}"

    # TODO add function call pre-process section

    return completion_text


@router.post("/v1/chat/completions", tags=["Completions"])
@router.post("/chat/completions", tags=["Completions"])
async def chat_completions(body: ChatCompletionBody, request: Request):
    model: TextRWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.messages is None or body.messages == []:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "messages not found")

    interface = model.interface
    user = model.user if body.user_name is None else body.user_name
    bot = model.bot if body.assistant_name is None else body.assistant_name

    if model.version < 5:
        completion_text = chat_template_old(model, body, interface, user, bot)
    else:
        completion_text = chat_template(model, body, interface, user, bot)

    user_code = model.pipeline.decode([model.pipeline.encode(user)[0]])
    bot_code = model.pipeline.decode([model.pipeline.encode(bot)[0]])
    if type(body.stop) == str:
        body.stop = [body.stop, f"\n\n{user_code}", f"\n\n{bot_code}"]
    elif type(body.stop) == list:
        body.stop.append(f"\n\n{user_code}")
        body.stop.append(f"\n\n{bot_code}")
    elif body.stop is None:
        body.stop = default_stop + [f"\n\n{user_code}", f"\n\n{bot_code}"]
    # if not body.presystem:
    #     body.stop.append("\n\n")

    if body.tool_choice != "none" and body.tools is not None and len(body.tools) > 0:
        return await chat_with_tools(model, body, request, completion_text)
    else:
        return await chat(model, body, request, completion_text)


async def chat_with_tools(
    model: TextRWKV, body: ChatCompletionBody, request: Request, completion_text: str
):
    system = "System" if body.system_name is None else body.system_name
    interface = model.interface
    tools_text = str((await request.json())["tools"])
    if tools_text == "[]":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unecepted tools input")
    
    # Prompts
    tools_text = \
f"""\
{system}{interface} there is a function list, you should chose one function which can resolve user's requirement,\
then fill the name and arguments.
function list: {tools_text}
e.g.:
User: <content>
Assistant: {{"name": "<name of the function you chose>", "arguments": '{{"<pram1>": "<arg1>", "<pram2>": "<arg2>", ...}}'}}
"""

    completion_text = tools_text + completion_text # TODO Tools
    response = await chat(model, body, request, completion_text)
    # TODO response = postprocess_response(response, ...)
    return response


async def chat(
    model: TextRWKV, body: ChatCompletionBody, request: Request, completion_text: str
):
    if body.stream:
        return EventSourceResponse(
            eval_rwkv(
                model, request, body, completion_text, body.stream, body.stop, True
            )
        )
    else:
        try:
            return await eval_rwkv(
                model, request, body, completion_text, body.stream, body.stop, True
            ).__anext__()
        except StopAsyncIteration:
            return None


@router.post("/v1/completions", tags=["Completions"])
@router.post("/completions", tags=["Completions"])
async def completions(body: CompletionBody, request: Request):
    model: AbstractRWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.prompt is None or body.prompt == "" or body.prompt == []:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "prompt not found")

    if type(body.prompt) == list:
        body.prompt = body.prompt[0]  # TODO: support multiple prompts

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
    input: Union[str, List[str], List[List[int]], None]
    model: Union[str, None] = "rwkv"
    encoding_format: str = None
    fast_mode: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "input": "a big apple",
                "model": "rwkv",
                "encoding_format": None,
                "fast_mode": False,
            }
        }
    }


def embedding_base64(embedding: List[float]) -> str:
    import numpy as np

    return base64.b64encode(np.array(embedding).astype(np.float32)).decode("utf-8")


@router.post("/v1/embeddings", tags=["Embeddings"])
@router.post("/embeddings", tags=["Embeddings"])
@router.post("/v1/engines/text-embedding-ada-002/embeddings", tags=["Embeddings"])
@router.post("/engines/text-embedding-ada-002/embeddings", tags=["Embeddings"])
async def embeddings(body: EmbeddingsBody, request: Request):
    model: AbstractRWKV = global_var.get(global_var.Model)
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
        with completion_lock:
            if await request.is_disconnected():
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
                    encoding = tiktoken.model.encoding_for_model(
                        "text-embedding-ada-002"
                    )
                    for i in range(len(body.input)):
                        if await request.is_disconnected():
                            break
                        input = encoding.decode(body.input[i])
                        embedding, token_len = model.get_embedding(
                            input, body.fast_mode
                        )
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
                embedding, prompt_tokens = model.get_embedding(
                    body.input, body.fast_mode
                )
                if base64_format:
                    embedding = embedding_base64(embedding)
                embeddings.append(embedding)

            requests_num = requests_num - 1
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
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens,
                },
            }