<p align="center">
    <img src="https://github.com/josStorer/RWKV-Runner/assets/13366013/65c46133-7506-4b54-b64f-fe49f188afa7">
</p>

<h1 align="center">RWKV Runner</h1>

<div align="center">

本项目旨在消除大语言模型的使用门槛，全自动为你处理一切，你只需要一个仅仅几MB的可执行程序。此外本项目提供了与OpenAI
API兼容的接口，这意味着一切ChatGPT客户端都是RWKV客户端。

[![license][license-image]][license-url]
[![release][release-image]][release-url]
[![py-version][py-version-image]][py-version-url]

[English](README.md) | 简体中文 | [日本語](README_JA.md)

### 安装

[![Windows][Windows-image]][Windows-url]
[![MacOS][MacOS-image]][MacOS-url]
[![Linux][Linux-image]][Linux-url]

[RWKV官方文档](https://rwkv.cn/docs) | [视频演示](https://www.bilibili.com/video/BV1hM4y1v76R) | [疑难解答](https://www.bilibili.com/read/cv23921171) | [预览](#Preview) | [下载][download-url] | [懒人包](https://pan.baidu.com/s/1zdzZ_a0uM3gDqi6pXIZVAA?pwd=1111) | [简明服务部署示例](#Simple-Deploy-Example) | [服务器部署示例](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples) | [MIDI硬件输入](#MIDI-Input)

[license-image]: http://img.shields.io/badge/license-MIT-blue.svg

[license-url]: https://github.com/josStorer/RWKV-Runner/blob/master/LICENSE

[release-image]: https://img.shields.io/github/release/josStorer/RWKV-Runner.svg

[release-url]: https://github.com/josStorer/RWKV-Runner/releases/latest

[py-version-image]: https://img.shields.io/pypi/pyversions/fastapi.svg

[py-version-url]: https://github.com/josStorer/RWKV-Runner/tree/master/backend-python

[download-url]: https://github.com/josStorer/RWKV-Runner/releases

[Windows-image]: https://img.shields.io/badge/-Windows-blue?logo=windows

[Windows-url]: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt

[MacOS-image]: https://img.shields.io/badge/-MacOS-black?logo=apple

[MacOS-url]: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt

[Linux-image]: https://img.shields.io/badge/-Linux-black?logo=linux

[Linux-url]: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt

</div>

## 小贴士

- 你可以在服务器部署[backend-python](./backend-python/)，然后将此程序仅用作客户端，在设置的`API URL`中填入你的服务器地址

- 如果你正在部署并对外提供公开服务，请通过API网关限制请求大小，避免过长的prompt提交占用资源。此外，请根据你的实际情况，限制请求的
  max_tokens 上限: https://github.com/josStorer/RWKV-Runner/blob/master/backend-python/utils/rwkv.py#L567,
  默认le=102400, 这可能导致极端情况下单个响应消耗大量资源

- 预设配置已经开启自定义CUDA算子加速，速度更快，且显存消耗更少。如果你遇到可能的兼容性(输出乱码)
  问题，前往配置页面，关闭`使用自定义CUDA算子加速`，或更新你的显卡驱动

- 如果 Windows Defender
  说这是一个病毒，你可以尝试下载[v1.3.7_win.zip](https://github.com/josStorer/RWKV-Runner/releases/download/v1.3.7/RWKV-Runner_win.zip)，
  然后让其自动更新到最新版，或添加信任 (`Windows Security` -> `Virus & threat protection` -> `Manage settings` -> `Exclusions` -> `Add or remove exclusions` -> `Add an exclusion` -> `Folder` -> `RWKV-Runner`)

- 对于不同的任务，调整API参数会获得更好的效果，例如对于翻译任务，你可以尝试设置Temperature为1，Top_P为0.3

## 功能

- RWKV模型管理，一键启动
- 前后端分离，如果你不想使用客户端，也允许单独部署前端服务，或后端推理服务，或具有WebUI的后端推理服务。
  [简明服务部署示例](#Simple-Deploy-Example) | [服务器部署示例](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples)
- 与OpenAI API兼容，一切ChatGPT客户端，都是RWKV客户端。启动模型后，打开 http://127.0.0.1:8000/docs 查看API文档
- 全自动依赖安装，你只需要一个轻巧的可执行程序
- 预设多级显存配置，几乎在各种电脑上工作良好。通过配置页面切换Strategy到WebGPU，还可以在AMD，Intel等显卡上运行
- 自带用户友好的聊天，续写，作曲交互页面。支持聊天预设，附件上传，MIDI硬件输入及音轨编辑。
  [预览](#Preview) | [MIDI硬件输入](#MIDI-Input)
- 内置WebUI选项，一键启动Web服务，共享硬件资源
- 易于理解和操作的参数配置，及各类操作引导提示
- 内置模型转换工具
- 内置下载管理和远程模型检视
- 内置一键LoRA微调 (仅限Windows)
- 也可用作 OpenAI ChatGPT, GPT Playground, Ollama 等服务的客户端 (在设置内填写API URL和API Key)
- 多语言本地化
- 主题切换
- 自动更新

## Simple Deploy Example

```bash
git clone https://github.com/josStorer/RWKV-Runner

# 然后
cd RWKV-Runner
python ./backend-python/main.py #后端推理服务已启动, 调用/switch-model载入模型, 参考API文档: http://127.0.0.1:8000/docs

# 或者
cd RWKV-Runner/frontend
npm ci
npm run build #编译前端
cd ..
python ./backend-python/webui_server.py #单独启动前端服务
# 或者
python ./backend-python/main.py --webui #同时启动前后端服务

# 帮助参数
python ./backend-python/main.py -h
```

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

注意: 1.4.0 版本对embeddings API质量进行了改善，生成结果与之前的版本不兼容，如果你正在使用此API生成知识库等，请重新生成

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

## MIDI Input

小贴士: 你可以下载 https://github.com/josStorer/sgm_plus, 并解压到程序的`assets/sound-font`目录, 以使用离线音源. 注意,
如果你正在从源码编译程序, 请不要将其放置在源码目录中

如果你没有MIDI键盘, 你可以使用像 `Virtual Midi Controller 3 LE` 这样的虚拟MIDI输入软件,
配合[loopMIDI](https://www.tobias-erichsen.de/wp-content/uploads/2020/01/loopMIDISetup_1_0_16_27.zip), 使用普通电脑键盘作为MIDI输入

### USB MIDI 连接

- USB MIDI设备是即插即用的, 你能够在作曲页面选择你的输入设备
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/a448c34a-56d8-46eb-8dc2-dd11e8e0c4ce)

### Mac MIDI 蓝牙连接

- 对于想要使用蓝牙输入的Mac用户,
  请安装[Bluetooth MIDI Connect](https://apps.apple.com/us/app/bluetooth-midi-connect/id1108321791), 启动后点击托盘连接,
  之后你可以在作曲页面选择你的输入设备
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c079a109-1e3d-45c1-bbf5-eed85da1550e)

### Windows MIDI 蓝牙连接

- Windows似乎只为UWP实现了蓝牙MIDI支持, 因此需要多个步骤进行连接, 我们需要创建一个本地的虚拟MIDI设备, 然后启动一个UWP应用,
  通过此UWP应用将蓝牙MIDI输入重定向到虚拟MIDI设备, 然后本软件监听虚拟MIDI设备的输入
- 因此, 首先你需要下载[loopMIDI](https://www.tobias-erichsen.de/wp-content/uploads/2020/01/loopMIDISetup_1_0_16_27.zip),
  用于创建虚拟MIDI设备, 点击左下角的加号创建设备
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/b75998ff-115c-4ddd-b97c-deeb5c106255)
- 然后, 你需要下载[Bluetooth LE Explorer](https://apps.microsoft.com/detail/9N0ZTKF1QD98), 以发现并连接蓝牙MIDI设备,
  点击Start搜索设备, 然后点击Pair绑定MIDI设备
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c142c3ea-a973-4531-9807-4c385d640a2b)
- 最后, 你需要安装[MIDIberry](https://apps.microsoft.com/detail/9N39720H2M05), 这个UWP应用能将MIDI蓝牙输入重定向到虚拟MIDI设备,
  启动后, 在输入栏, 双击你实际的蓝牙MIDI设备名称, 在输出栏, 双击我们先前创建的虚拟MIDI设备名称
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/5ad6a1d9-4f68-4d95-ae17-4296107d1669)
- 现在, 你可以在作曲页面选择虚拟MIDI设备作为输入. Bluetooth LE Explorer不再需要运行, loopMIDI窗口也可以退出, 它会自动在后台运行,
  仅保持MIDIberry打开即可
- ![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/6460c355-884e-4b28-a2eb-8ab7a2e3a01a)

## 相关仓库:

- RWKV-5-World: https://huggingface.co/BlinkDL/rwkv-5-world/tree/main
- RWKV-4-World: https://huggingface.co/BlinkDL/rwkv-4-world/tree/main
- RWKV-4-Raven: https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main
- ChatRWKV: https://github.com/BlinkDL/ChatRWKV
- RWKV-LM: https://github.com/BlinkDL/RWKV-LM
- RWKV-LM-LoRA: https://github.com/Blealtan/RWKV-LM-LoRA
- RWKV-v5-lora: https://github.com/JL-er/RWKV-v5-lora
- MIDI-LLM-tokenizer: https://github.com/briansemrau/MIDI-LLM-tokenizer
- ai00_rwkv_server: https://github.com/cgisky1980/ai00_rwkv_server
- rwkv.cpp: https://github.com/saharNooby/rwkv.cpp
- web-rwkv-py: https://github.com/cryscan/web-rwkv-py
- web-rwkv: https://github.com/cryscan/web-rwkv

## Preview

### 主页

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/3265b11a-ab19-4e19-bfea-fc687f59aaf9)

### 聊天

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/9570e73b-dca2-4316-9e92-09961f3c48c4)

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/162fce43-8568-4850-a6af-ab60af988da6)

### 续写

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/69f9ba7a-2fe8-4a5e-94cb-aa655aa409e2)

### 作曲

小贴士: 你可以下载 https://github.com/josStorer/sgm_plus, 并解压到程序的`assets/sound-font`目录, 以使用离线音源. 注意,
如果你正在从源码编译程序, 请不要将其放置在源码目录中

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/95b34893-80c2-4706-87f9-bc141032ed4b)

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/3cb31ca8-d708-42f1-8768-1605fb0b2174)

### 配置

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/0f4d4f21-8abe-4f4d-8c4f-cd7d5607f20e)

### 模型管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/871f2d2a-7e41-4be7-9b32-be1b3e00dc3e)

### 下载管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/cc076038-2a91-4d36-bd39-266020e8ea87)

### LoRA微调

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/31939b8f-9546-4f44-b434-295b492ec625)

### 设置

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/9652d7cc-ac33-4587-a8fb-03e5a6f5ea77)
