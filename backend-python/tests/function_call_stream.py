# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/guides/chat
import time
import json
from openai import OpenAI
from collections import defaultdict

# record the time before the request is sent
start_time = time.time()


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


client = OpenAI(
    base_url="http://127.0.0.1:8000",
    api_key="test",
)

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Paris?",
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
    stream=True,
)

# https://community.openai.com/t/has-anyone-managed-to-get-a-tool-call-working-when-stream-true/498867/11
tool_calls = []
index = 0
start = True
for chunk in response:
    print(chunk)
    chunk_time = time.time() - start_time

    delta = chunk.choices[0].delta
    if not delta:
        break
    if not delta.function_call and not delta.tool_calls:
        if start:
            continue
        else:
            break
    start = False
    if delta.function_call:
        if index == len(tool_calls):
            tool_calls.append(defaultdict(str))
        if delta.function_call.name:
            tool_calls[index]["name"] = delta.function_call.name
        if delta.function_call.arguments:
            tool_calls[index]["arguments"] += delta.function_call.arguments
    elif delta.tool_calls:
        tool_call = delta.tool_calls[0]
        index = tool_call.index
        if index == len(tool_calls):
            tool_calls.append(defaultdict(str))
        if tool_call.id:
            tool_calls[index]["id"] = tool_call.id
        if tool_call.function:
            if tool_call.function.name:
                tool_calls[index]["name"] = tool_call.function.name
            if tool_call.function.arguments:
                tool_calls[index]["arguments"] += tool_call.function.arguments

print()
print(tool_calls)
print(f"Full response received {chunk_time:.2f} seconds after request")

if tool_calls:
    # Step 3: call the function
    # Note: the JSON response may not always be valid; be sure to handle errors
    available_functions = {
        "get_current_weather": get_current_weather,
    }  # only one function in this example, but you can have multiple
    # Step 4: send the info for each function call and function response to the model
    for tool_call in tool_calls:
        function_name = tool_call["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call["arguments"])
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": tool_call["arguments"],
                        },
                    }
                ],
            }
        )  # extend conversation with assistant's reply
        messages.append(
            {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": function_response,
            }
        )  # extend conversation with function response
    second_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )  # get a new response from the model where it can see the function response
    print(second_response.choices[0].message.content)
