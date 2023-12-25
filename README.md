<p align="center">
    <img src="https://github.com/josStorer/RWKV-Runner/assets/13366013/d24834b0-265d-45f5-93c0-fac1e19562af">
</p>

<h1 align="center">RWKV Runner</h1>

<div align="center">

This project aims to eliminate the barriers of using large language models by automating everything for you. All you
need is a lightweight executable program of just a few megabytes. Additionally, this project provides an interface
compatible with the OpenAI API, which means that every ChatGPT client is an RWKV client.

[![license][license-image]][license-url]
[![release][release-image]][release-url]

English | [简体中文](README_ZH.md) | [日本語](README_JA.md)

### Install

[![Windows][Windows-image]][Windows-url]
[![MacOS][MacOS-image]][MacOS-url]
[![Linux][Linux-image]][Linux-url]

[FAQs](https://github.com/josStorer/RWKV-Runner/wiki/FAQs) | [Preview](#Preview) | [Download][download-url] | [Simple Deploy Example](#Simple-Deploy-Example) | [Server Deploy Examples](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples) | [MIDI Hardware Input](#MIDI-Input)

[license-image]: http://img.shields.io/badge/license-MIT-blue.svg

[license-url]: https://github.com/josStorer/RWKV-Runner/blob/master/LICENSE

[release-image]: https://img.shields.io/github/release/josStorer/RWKV-Runner.svg

[release-url]: https://github.com/josStorer/RWKV-Runner/releases/latest

[download-url]: https://github.com/josStorer/RWKV-Runner/releases

[Windows-image]: https://img.shields.io/badge/-Windows-blue?logo=windows

[Windows-url]: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt

[MacOS-image]: https://img.shields.io/badge/-MacOS-black?logo=apple

[MacOS-url]: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt

[Linux-image]: https://img.shields.io/badge/-Linux-black?logo=linux

[Linux-url]: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt

</div>

## Tips

- You can deploy [backend-python](./backend-python/) on a server and use this program as a client only. Fill in
  your server address in the Settings `API URL`.

- If you are deploying and providing public services, please limit the request size through API gateway to prevent
  excessive resource usage caused by submitting overly long prompts. Additionally, please restrict the upper limit of
  requests' max_tokens based on your actual
  situation: https://github.com/josStorer/RWKV-Runner/blob/master/backend-python/utils/rwkv.py#L567, the default is set
  as le=102400, which may result in significant resource consumption for individual responses in extreme cases.

- Default configs has enabled custom CUDA kernel acceleration, which is much faster and consumes much less VRAM. If you
  encounter possible compatibility issues (output garbled), go to the Configs page and turn
  off `Use Custom CUDA kernel to Accelerate`, or try to upgrade your gpu driver.

- If Windows Defender claims this is a virus, you can try
  downloading [v1.3.7_win.zip](https://github.com/josStorer/RWKV-Runner/releases/download/v1.3.7/RWKV-Runner_win.zip)
  and letting it update automatically to the latest version, or add it to the trusted
  list (`Windows Security` -> `Virus & threat protection` -> `Manage settings` -> `Exclusions` -> `Add or remove exclusions` -> `Add an exclusion` -> `Folder` -> `RWKV-Runner`).

- For different tasks, adjusting API parameters can achieve better results. For example, for translation tasks, you can
  try setting Temperature to 1 and Top_P to 0.3.

## Features

- RWKV model management and one-click startup.
- Front-end and back-end separation, if you don't want to use the client, also allows for separately deploying the
  front-end service, or the back-end inference service, or the back-end inference service with a WebUI.
  [Simple Deploy Example](#Simple-Deploy-Example) | [Server Deploy Examples](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples)
- Compatible with the OpenAI API, making every ChatGPT client an RWKV client. After starting the model,
  open http://127.0.0.1:8000/docs to view more details.
- Automatic dependency installation, requiring only a lightweight executable program.
- Pre-set multi-level VRAM configs, works well on almost all computers. In Configs page, switch Strategy to WebGPU, it
  can also run on AMD, Intel, and other graphics cards.
- User-friendly chat, completion, and composition interaction interface included. Also supports chat presets, attachment
  uploads, MIDI hardware input, and track editing.
  [Preview](#Preview) | [MIDI Hardware Input](#MIDI-Input)
- Built-in WebUI option, one-click start of Web service, sharing your hardware resources.
- Easy-to-understand and operate parameter configuration, along with various operation guidance prompts.
- Built-in model conversion tool.
- Built-in download management and remote model inspection.
- Built-in one-click LoRA Finetune. (Windows Only)
- Can also be used as an OpenAI ChatGPT and GPT-Playground client. (Fill in the API URL and API Key in Settings page)
- Multilingual localization.
- Theme switching.
- Automatic updates.

## Simple Deploy Example

```bash
git clone https://github.com/josStorer/RWKV-Runner

# Then
cd RWKV-Runner
python ./backend-python/main.py #The backend inference service has been started, request /switch-model API to load the model, refer to the API documentation: http://127.0.0.1:8000/docs

# Or
cd RWKV-Runner/frontend
npm ci
npm run build #Compile the frontend
cd ..
python ./backend-python/webui_server.py #Start the frontend service separately
# Or
python ./backend-python/main.py --webui #Start the frontend and backend service at the same time

# Help Info
python ./backend-python/main.py -h
```

## API Concurrency Stress Testing

```bash
ab -p body.json -T application/json -c 20 -n 100 -l http://127.0.0.1:8000/chat/completions
```

body.json:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    }
  ]
}
```

## Embeddings API Example

Note: v1.4.0 has improved the quality of embeddings API. The generated results are not compatible
with previous versions. If you are using embeddings API to generate knowledge bases or similar, please regenerate.

If you are using langchain, just use `OpenAIEmbeddings(openai_api_base="http://127.0.0.1:8000", openai_api_key="sk-")`

```python
import numpy as np
import requests


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


values = [
    "I am a girl",
    "我是个女孩",
    "私は女の子です",
    "广东人爱吃福建人",
    "我是个人类",
    "I am a human",
    "that dog is so cute",
    "私はねこむすめです、にゃん♪",
    "宇宙级特大事件！号外号外！"
]

embeddings = []
for v in values:
    r = requests.post("http://127.0.0.1:8000/embeddings", json={"input": v})
    embedding = r.json()["data"][0]["embedding"]
    embeddings.append(embedding)

compared_embedding = embeddings[0]

embeddings_cos_sim = [cosine_similarity(compared_embedding, e) for e in embeddings]

for i in np.argsort(embeddings_cos_sim)[::-1]:
    print(f"{embeddings_cos_sim[i]:.10f} - {values[i]}")
```

## MIDI Input

Tip: You can download https://github.com/josStorer/sgm_plus and unzip it to the program's `assets/sound-font` directory
to use it as an offline sound source. Please note that if you are compiling the program from source code, do not place
it in the source code directory.

If you don't have a MIDI keyboard, you can use virtual MIDI input software like `Virtual Midi Controller 3 LE`, along
with [loopMIDI](https://www.tobias-erichsen.de/wp-content/uploads/2020/01/loopMIDISetup_1_0_16_27.zip), to use a regular
computer keyboard as MIDI input.

### USB MIDI Connection

- USB MIDI devices are plug-and-play, and you can select your input device in the Composition page
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/13bb92c3-4504-482d-ab82-026ac6c31095)

### Mac MIDI Bluetooth Connection

- For Mac users who want to use Bluetooth input,
  please install [Bluetooth MIDI Connect](https://apps.apple.com/us/app/bluetooth-midi-connect/id1108321791), then click
  the tray icon to connect after launching,
  afterwards, you can select your input device in the Composition page.
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c079a109-1e3d-45c1-bbf5-eed85da1550e)

### Windows MIDI Bluetooth Connection

- Windows seems to have implemented Bluetooth MIDI support only for UWP (Universal Windows Platform) apps. Therefore, it
  requires multiple steps to establish a connection. We need to create a local virtual MIDI device and then launch a UWP
  application. Through this UWP application, we will redirect Bluetooth MIDI input to the virtual MIDI device, and then
  this software will listen to the input from the virtual MIDI device.
- So, first, you need to
  download [loopMIDI](https://www.tobias-erichsen.de/wp-content/uploads/2020/01/loopMIDISetup_1_0_16_27.zip)
  to create a virtual MIDI device. Click the plus sign in the bottom left corner to create the device.
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/b75998ff-115c-4ddd-b97c-deeb5c106255)
- Next, you need to download [Bluetooth LE Explorer](https://apps.microsoft.com/detail/9N0ZTKF1QD98) to discover and
  connect to Bluetooth MIDI devices. Click "Start" to search for devices, and then click "Pair" to bind the MIDI device.
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c142c3ea-a973-4531-9807-4c385d640a2b)
- Finally, you need to install [MIDIberry](https://apps.microsoft.com/detail/9N39720H2M05),
  This UWP application can redirect Bluetooth MIDI input to the virtual MIDI device. After launching it, double-click
  your actual Bluetooth MIDI device name in the input field, and in the output field, double-click the virtual MIDI
  device name we created earlier.
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/5ad6a1d9-4f68-4d95-ae17-4296107d1669)
- Now, you can select the virtual MIDI device as the input in the Composition page. Bluetooth LE Explorer no longer
  needs to run, and you can also close the loopMIDI window, it will run automatically in the background. Just keep
  MIDIberry open.
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/1c371821-c7b7-4c18-8e42-9e315efbe427)

## Related Repositories:

- RWKV-5-World: https://huggingface.co/BlinkDL/rwkv-5-world/tree/main
- RWKV-4-World: https://huggingface.co/BlinkDL/rwkv-4-world/tree/main
- RWKV-4-Raven: https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main
- ChatRWKV: https://github.com/BlinkDL/ChatRWKV
- RWKV-LM: https://github.com/BlinkDL/RWKV-LM
- RWKV-LM-LoRA: https://github.com/Blealtan/RWKV-LM-LoRA
- MIDI-LLM-tokenizer: https://github.com/briansemrau/MIDI-LLM-tokenizer
- ai00_rwkv_server: https://github.com/cgisky1980/ai00_rwkv_server
- rwkv.cpp: https://github.com/saharNooby/rwkv.cpp
- web-rwkv-py: https://github.com/cryscan/web-rwkv-py

## Preview

### Homepage

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c9b9cdd0-63f9-4319-9f74-5bf5d7df5a67)

### Chat

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/80009872-528f-4932-aeb2-f724fa892e7c)

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/e98c9038-3323-47b0-8edb-d639fafd37b2)

### Completion

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/bf49de8e-3b89-4543-b1ef-7cd4b19a1836)

### Composition

Tip: You can download https://github.com/josStorer/sgm_plus and unzip it to the program's `assets/sound-font` directory
to use it as an offline sound source. Please note that if you are compiling the program from source code, do not place
it in the source code directory.

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/e8ad908d-3fd2-4e92-bcdb-96815cb836ee)

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/b2ce4761-9e75-477e-a182-d0255fb8ac76)

### Configuration

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/f41060dc-5517-44af-bb3f-8ef71720016d)

### Model Management

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/b1581147-a6ce-4493-8010-e33c0ddeca0a)

### Download Management

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c8153cf9-c8cb-4618-8268-60c82a5be539)

### LoRA Finetune

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/4715045a-683e-4d2a-9b0e-090c7a5df63f)

### Settings

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/1067e635-8c07-4217-86a8-e48a5fcbb075)
