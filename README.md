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

English | [简体中文](README_ZH.md)

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

#### Default configs has enabled custom CUDA kernel acceleration, which is much faster and consumes much less VRAM. If you encounter possible compatibility issues, go to the Configs page and turn off `Use Custom CUDA kernel to Accelerate`.

#### If Windows Defender claims this is a virus, you can try downloading [v1.0.8](https://github.com/josStorer/RWKV-Runner/releases/tag/v1.0.8)/[v1.0.9](https://github.com/josStorer/RWKV-Runner/releases/tag/v1.0.9) and letting it update automatically to the latest version, or add it to the trusted list.

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

## Todo

- [ ] Model training functionality
- [x] CUDA operator int8 acceleration
- [x] macOS support
- [x] Linux support
- [ ] Local State Cache DB

## Related Repositories:

- RWKV-4-Raven: https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main
- ChatRWKV: https://github.com/BlinkDL/ChatRWKV
- RWKV-LM: https://github.com/BlinkDL/RWKV-LM

## Preview

### Homepage

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/60efbb65-29e3-4346-a597-5bdcd099251c)

### Chat

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/6cde9c45-51bb-4dee-b1fe-746862448520)

### Completion

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/52f47f92-d21d-4cd7-b04e-d6f9af937a97)

### Configuration

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/93270a68-9d6d-4247-b6a3-e543c65a876b)

### Model Management

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/6f96fdd3-fdf5-4b78-af80-2afbd1ad173b)

### Download Management

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/6982e7ee-bace-4a88-bb47-92379185bf9d)

### Settings

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/b3b2ab46-344c-4f04-b066-1503f776eeb9)
