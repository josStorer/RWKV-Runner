# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/guides/chat
import time
from openai import OpenAI

# record the time before the request is sent
start_time = time.time()

client = OpenAI(
    base_url="http://127.0.0.1:8000",
    api_key="test",
)

messages = [
        {
            "role": "user",
            "content": "Hello!",
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

# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []

# iterate through the stream of events
for chunk in response:
    chunk_time = time.time() - start_time  # calculate the time delay of the chunk
    collected_chunks.append(chunk)  # save the event response
    chunk_message = chunk.choices[0].delta.content  # extract the message
    collected_messages.append(chunk_message)  # save the message
    print(chunk_message, end='')

print()
print(f"Full response received {chunk_time:.2f} seconds after request")
