import asyncio
import json
from threading import Lock
from typing import List, Union, Literal
from enum import Enum
import base64
import time, re, random, string

from fastapi import APIRouter, Request, status, HTTPException
from fastapi.encoders import jsonable_encoder
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
    "\n\nObservation",
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
    body: ModelConfigBody,
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
            try:
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
            except:
                pass
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
            else:  # !stream
                yield {
                    "object": "chat.completion" if chat_mode else "text_completion",
                    "model": model.name,
                    "choices": [
                        (
                            {
                                "message": {
                                    "role": Role.Assistant.value,
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
                        )
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
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
    tool = "Observation"
    for message in body.messages:
        append_message: str = ""
        if message.role == Role.User.value:
            append_message = f"{user}{interface} " + message.content
        elif message.role == Role.Assistant.value:
            if message.tool_calls and len(message.tool_calls) > 0:
                name = message.tool_calls[0].function.name
                arguments = json.loads(message.tool_calls[0].function.arguments)
                arguments = ", ".join([f'"{k}"="{v}"' for k, v in arguments.items()])
                append_message = (
                    f"{bot}{interface} "
                    + f"{name}\n```python\ntool_call({arguments})\n```"
                )
            elif message.content is None:
                continue
            else:
                append_message = f"{bot}{interface} " + message.content
        elif message.role == Role.System.value:
            append_message = f"{system}{interface} " + message.content
        elif message.role == Role.Tool.value:
            append_message = f"{tool}{interface} " + message.content
        completion_text += append_message + "\n\n"
    completion_text += f"{bot}{interface}"
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

    if (
        body.tool_choice != "none" and body.tools is not None and len(body.tools) > 0
    ) or body.messages[-1].role == Role.Tool.value:
        return await chat_with_tools(model, body, request, completion_text)
    else:
        return await chat(model, body, request, completion_text)


tool_call_id_timestamps = {}


async def chat_with_tools(
    model: TextRWKV, body: ChatCompletionBody, request: Request, completion_text: str
):
    system = "System"
    interface = model.interface
    is_with_tool_call_id = body.messages[-1].role == Role.Tool.value
    if is_with_tool_call_id:
        tool_call_id = body.messages[-1].tool_call_id
        tools_text = tool_call_id_timestamps.get(tool_call_id)
    else:
        tools = [tool.function for tool in body.tools]
        tools_text = json.dumps(jsonable_encoder(tools), indent=2)
        tool_call_id = generate_tool_call_id()
        tool_call_id_timestamps[tool_call_id] = tools_text
        if len(tool_call_id_timestamps) > 1000:
            tool_call_id_timestamps.pop(next(iter(tool_call_id_timestamps)))

    # Function Call Prompts
    tools_text = f"""\
{system}{interface} You are a helpful assistant with access to the following functions. Use them if required -{tools_text}
"""

    completion_text = tools_text + "\n" + completion_text

    if is_with_tool_call_id:
        return await chat(model, body, request, completion_text)
    if body.stream:
        response = async_generator_stream_response_tool_call(
            model, body, request, completion_text, tool_call_id
        )
        return EventSourceResponse(response)
    else:
        response = await chat(model, body, request, completion_text)
        if response is not None:
            response = postprocess_response(response, tool_call_id)
        return response


def generate_tool_call_id():
    return "call_" + "".join(random.sample(string.ascii_letters + string.digits, 24))


async def async_generator_stream_response_tool_call(
    model: TextRWKV,
    body: ChatCompletionBody,
    request: Request,
    completion_text: str,
    tool_call_id: str,
):
    # NOTE: There is none of existing failure analysis.

    # Initialization
    gen = eval_rwkv(
        model, request, body, completion_text, body.stream, body.stop, True
    )  # Get an async generator handle
    content: str = ""
    flag_is_function_call_confirmed = False
    flag_is_common_confirmed = False
    convert_equal_to_colon = False

    # Loop, there is only one existing endpoint.
    done = False
    stack_keyword_pairs = [["```", "```"], ["(", ")"], ['"', '"'], ["'", "'"]]
    while True:
        if done:
            if not flag_is_common_confirmed and not flag_is_function_call_confirmed:
                yield json.dumps(
                    {
                        "object": "chat.completion.chunk",
                        "model": model.name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                yield json.dumps(
                    {
                        "object": "chat.completion.chunk",
                        "model": model.name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
            elif flag_is_function_call_confirmed:
                yield json.dumps(
                    {
                        "object": "chat.completion.chunk",
                        "model": model.name,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                        ],
                    }
                )
            yield "[DONE]"
            break

        try:
            response = await gen.__anext__()  # Generate a delta response
            if response == "[DONE]":
                done = True
                continue
        except StopAsyncIteration:
            break

        if flag_is_common_confirmed:
            yield response
            continue

        # Post process response
        response_decoded = json.loads(response)  # Decode string
        delta = response_decoded["choices"][0]["delta"]
        if delta == {}:
            continue
        delta_content: str = delta["content"]
        content += delta_content

        if flag_is_function_call_confirmed:
            if "\n\n" in content:
                done = True
                continue

            for pair in stack_keyword_pairs:
                if done:
                    break
                for keyword in pair:
                    if keyword in delta_content:
                        stack.append(keyword)
                        if (
                            pair[0] in stack
                            and pair[1] in stack
                            and (
                                (
                                    pair[0] != pair[1]
                                    and stack.index(pair[0]) < stack.index(pair[1])
                                )
                                or (pair[0] == pair[1] and stack.count(pair[0]) >= 2)
                            )
                        ):
                            stack.remove(pair[0])
                            stack.remove(pair[1])
                            if pair[0] == '"' or pair[0] == "'":
                                convert_equal_to_colon = True
                            if "(" not in stack and ")" not in stack:
                                done = True
                                response_decoded["choices"][0]["delta"] = {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "arguments": (
                                                    '"'
                                                    if delta_content.strip().startswith(
                                                        '"'
                                                    )
                                                    else (
                                                        "'"
                                                        if delta_content.strip().startswith(
                                                            "'"
                                                        )
                                                        else ""
                                                    )
                                                )
                                                + "}",
                                            },
                                        }
                                    ]
                                }
                                yield json.dumps(response_decoded)
                                break
            if done:
                continue

            if "=" in delta_content and convert_equal_to_colon:
                delta_content = delta_content.replace("=", ":")
                convert_equal_to_colon = False
            response_decoded["choices"][0]["delta"]["content"] = None
            response_decoded["choices"][0]["delta"] = {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "arguments": delta_content,
                        },
                    }
                ]
            }
            yield json.dumps(response_decoded)
            continue

        if not flag_is_common_confirmed and not flag_is_function_call_confirmed:
            """
            # Unconfirmed Response, check content field by the followings:
            # Up to 4 line feeds:                                       Common Response.
            # Up to 60 characters:                                      Common Response.
            # Up to 44 characters under markdown code block unclosed:   Common Response.
            # Field "```FunctionName\ntool_call(...)```" detected:      Function Call Response.
            #                                                           - There will be 2 responses generated.
            # Default:                                                  Unsure Response.
            #                                                           - Recheck with the next delta.content field added.
            """
            # Constant
            LIMIT_LINE_FEEDS = 4
            LIMIT_CHARACTERS = 60
            LIMIT_FUNCTION_NAME_CHARACTERS = 44
            REGEX_BLOCKS_HEADERS = r"([\w]+)[\s]*```[\w\s]*tool_call\("

            # Regex
            regex_match_function_call_head: Union[re.Match, None] = re.search(
                REGEX_BLOCKS_HEADERS, content
            )

            # Confirm Common Response
            if regex_match_function_call_head is None and (
                content.count("\n") >= LIMIT_LINE_FEEDS
                or len(content) > LIMIT_CHARACTERS
                or (
                    len(content) > LIMIT_FUNCTION_NAME_CHARACTERS
                    and "```" not in content
                )
            ):
                flag_is_common_confirmed = True
                response_decoded["choices"][0]["delta"]["content"] = content
                yield json.dumps(response_decoded)
                continue

            # Confirm Function call Response
            if regex_match_function_call_head is not None:
                flag_is_function_call_confirmed = True
                stack = ["```", "("]

                # Generate a blank content response
                response_decoded["choices"][0]["delta"]["role"] = "assistant"
                response_decoded["choices"][0]["delta"]["content"] = None
                yield json.dumps(response_decoded)

                # Generate a function call details response
                name = regex_match_function_call_head.group(1)
                response_decoded["choices"][0]["delta"] = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": "",
                            },
                        }
                    ]
                }
                yield json.dumps(response_decoded)
                response_decoded["choices"][0]["delta"] = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": "{"
                                + (
                                    '"'
                                    if delta_content.strip().endswith('"')
                                    else (
                                        "'"
                                        if delta_content.strip().endswith("'")
                                        else ""
                                    )
                                ),
                            },
                        }
                    ]
                }
                yield json.dumps(response_decoded)
                continue

        # Default: Unsure Response
        continue
        # End of loop body


def postprocess_response(response: dict, tool_call_id: str):
    # NOTE: There is none of existing failure analysis.
    REGEX_BLOCKS = r"([\w]+)[\s]*```[\w\s]*tool_call(.*?)\n*```"
    REGEX_ARGS = r'[\'"]([^\'"]+)[\'"]\s*=\s*[\'"]([^\'"]+)[\'"]'

    regex_match = re.search(
        REGEX_BLOCKS, response["choices"][0]["message"]["content"], re.DOTALL
    )
    if regex_match is None:
        return response

    name = regex_match.group(1)
    function = regex_match.group(2).strip()
    try:
        arguments = json.dumps(dict(re.findall(REGEX_ARGS, function)))
    except:
        return response

    tool_calls = [
        {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
    ]

    response["choices"][0]["message"]["tool_calls"] = tool_calls
    response["choices"][0]["message"]["content"] = None
    response["choices"][0]["finish_reason"] = "tool_calls"

    return response


# -----------------------------------
# @Description: (reserved) post process multi-function-call responses
# -----------------------------------
# def postprocess_response(response: dict):
#     REGEX_BLOCKS = r'```[\w]*(.*?)```'
#     REGEX_FUNCTIONS = r'(\w+)*\('
#     REGEX_ARGS = r'"([^"]+)"\s*=\s*"([^"]+)"'

#     tool_calls = []
#     blocks = re.findall(REGEX_BLOCKS, response["choices"][0]["message"]["content"], re.DOTALL)
#     for block in blocks:
#         functions = block.strip().split('\n')
#         for function in functions:
#             name = re.search(REGEX_FUNCTIONS, function).group(1)
#             arguments = json.dumps(dict(re.findall(REGEX_ARGS, function)))
#             tool_calls.append(
#                 {
#                     "id": tool_call_id,
#                     "type": "function",
#                     "function": {
#                         "name": name,
#                         "arguments": arguments,
#                     }
#                 }
#             )

#     response["choices"][0]["message"]["tool_calls"] = tool_calls
#     response["choices"][0]["message"]["content"] = None
#     response["choices"][0]["finish_reason"] = "tool_calls"

#     return response


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
