<p align="center">
    <img src="https://github.com/josStorer/RWKV-Runner/assets/13366013/d24834b0-265d-45f5-93c0-fac1e19562af">
</p>

<h1 align="center">RWKV Runner</h1>

<div align="center">

本项目旨在消除大语言模型的使用门槛，全自动为你处理一切，你只需要一个仅仅几MB的可执行程序。此外本项目提供了与OpenAI
API兼容的接口，这意味着一切ChatGPT客户端都是RWKV客户端。

[![license][license-image]][license-url]
[![release][release-image]][release-url]

[English](README.md) | 简体中文 | [日本語](README_JA.md)

### 安装

[![Windows][Windows-image]][Windows-url]
[![MacOS][MacOS-image]][MacOS-url]
[![Linux][Linux-image]][Linux-url]

[视频演示](https://www.bilibili.com/video/BV1hM4y1v76R) | [疑难解答](https://www.bilibili.com/read/cv23921171) | [预览](#Preview) | [下载][download-url] | [懒人包](https://pan.baidu.com/s/1zdzZ_a0uM3gDqi6pXIZVAA?pwd=1111) | [服务器部署示例](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples)

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

#### 预设配置已经开启自定义CUDA算子加速，速度更快，且显存消耗更少。如果你遇到可能的兼容性问题，前往配置页面，关闭`使用自定义CUDA算子加速`

#### 如果Windows Defender说这是一个病毒，你可以尝试下载[v1.3.7_win.zip](https://github.com/josStorer/RWKV-Runner/releases/download/v1.3.7/RWKV-Runner_win.zip)，然后让其自动更新到最新版，或添加信任 (`Windows Security` -> `Virus & threat protection` -> `Manage settings` -> `Exclusions` -> `Add or remove exclusions` -> `Add an exclusion` -> `Folder` -> `RWKV-Runner`)

#### 对于不同的任务，调整API参数会获得更好的效果，例如对于翻译任务，你可以尝试设置Temperature为1，Top_P为0.3

## 功能

- RWKV模型管理，一键启动
- 与OpenAI API完全兼容，一切ChatGPT客户端，都是RWKV客户端。启动模型后，打开 http://127.0.0.1:8000/docs 查看详细内容
- 全自动依赖安装，你只需要一个轻巧的可执行程序
- 预设了2G至32G显存的配置，几乎在各种电脑上工作良好
- 自带用户友好的聊天和续写交互页面
- 易于理解和操作的参数配置
- 内置模型转换工具
- 内置下载管理和远程模型检视
- 内置一键LoRA微调
- 也可用作 OpenAI ChatGPT 和 GPT Playground 客户端
- 多语言本地化
- 主题切换
- 自动更新

## API并发压力测试

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

## Embeddings API 示例

如果你在用langchain, 直接使用 `OpenAIEmbeddings(openai_api_base="http://127.0.0.1:8000", openai_api_key="sk-")`

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

## 相关仓库:

- RWKV-4-World: https://huggingface.co/BlinkDL/rwkv-4-world/tree/main
- RWKV-4-Raven: https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main
- ChatRWKV: https://github.com/BlinkDL/ChatRWKV
- RWKV-LM: https://github.com/BlinkDL/RWKV-LM
- RWKV-LM-LoRA: https://github.com/Blealtan/RWKV-LM-LoRA

## Preview

### 主页

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/ff2b1eef-dd3b-4cbf-98fb-b5a1ecee43e1)

### 聊天

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/9570e73b-dca2-4316-9e92-09961f3c48c4)

### 续写

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/69f9ba7a-2fe8-4a5e-94cb-aa655aa409e2)

### 配置

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/59460f69-b172-4c7a-86cb-573262543076)

### 模型管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/551121ee-1bfe-421b-a9d1-24125126ab4b)

### 下载管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/cc076038-2a91-4d36-bd39-266020e8ea87)

### LoRA微调

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/31939b8f-9546-4f44-b434-295b492ec625)

### 设置

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/9652d7cc-ac33-4587-a8fb-03e5a6f5ea77)
