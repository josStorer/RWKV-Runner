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

[FAQs](https://github.com/josStorer/RWKV-Runner/wiki/FAQs) | [Preview](#Preview) | [Download][download-url] | [Server-Deploy-Examples](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples)

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

#### Tip: You can deploy [backend-python](./backend-python/) on a server and use this program as a client only. Fill in your server address in the Settings `API URL`.

#### Default configs has enabled custom CUDA kernel acceleration, which is much faster and consumes much less VRAM. If you encounter possible compatibility issues (output garbled), go to the Configs page and turn off `Use Custom CUDA kernel to Accelerate`, or try to upgrade your gpu driver.

#### If Windows Defender claims this is a virus, you can try downloading [v1.3.7_win.zip](https://github.com/josStorer/RWKV-Runner/releases/download/v1.3.7/RWKV-Runner_win.zip) and letting it update automatically to the latest version, or add it to the trusted list (`Windows Security` -> `Virus & threat protection` -> `Manage settings` -> `Exclusions` -> `Add or remove exclusions` -> `Add an exclusion` -> `Folder` -> `RWKV-Runner`).

#### For different tasks, adjusting API parameters can achieve better results. For example, for translation tasks, you can try setting Temperature to 1 and Top_P to 0.3.

## Features

- RWKV model management and one-click startup
- Fully compatible with the OpenAI API, making every ChatGPT client an RWKV client. After starting the model,
  open http://127.0.0.1:8000/docs to view more details.
- Automatic dependency installation, requiring only a lightweight executable program
- Configs with 2G to 32G VRAM are included, works well on almost all computers
- User-friendly chat and completion interaction interface included
- Easy-to-understand and operate parameter configuration
- Built-in model conversion tool
- Built-in download management and remote model inspection
- Built-in one-click LoRA Finetune
- Can also be used as an OpenAI ChatGPT and GPT-Playground client
- Multilingual localization
- Theme switching
- Automatic updates

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

## Related Repositories:

- RWKV-4-World: https://huggingface.co/BlinkDL/rwkv-4-world/tree/main
- RWKV-4-Raven: https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main
- ChatRWKV: https://github.com/BlinkDL/ChatRWKV
- RWKV-LM: https://github.com/BlinkDL/RWKV-LM
- RWKV-LM-LoRA: https://github.com/Blealtan/RWKV-LM-LoRA
- MIDI-LLM-tokenizer: https://github.com/briansemrau/MIDI-LLM-tokenizer

## Preview

### Homepage

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/d7f24d80-f382-428d-8b28-edf87e1549e2)

### Chat

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/80009872-528f-4932-aeb2-f724fa892e7c)

### Completion

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/bf49de8e-3b89-4543-b1ef-7cd4b19a1836)

### Composition

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/e8ad908d-3fd2-4e92-bcdb-96815cb836ee)

### Configuration

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/48befdc6-e03c-4851-9bee-22f77ee2640e)

### Model Management

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/367fe4f8-cc12-475f-9371-3cf62cdbf293)

### Download Management

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c8153cf9-c8cb-4618-8268-60c82a5be539)

### LoRA Finetune

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/4715045a-683e-4d2a-9b0e-090c7a5df63f)

### Settings

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/1067e635-8c07-4217-86a8-e48a5fcbb075)
