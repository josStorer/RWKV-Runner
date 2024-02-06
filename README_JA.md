<p align="center">
    <img src="https://github.com/josStorer/RWKV-Runner/assets/13366013/d24834b0-265d-45f5-93c0-fac1e19562af">
</p>

<h1 align="center">RWKV Runner</h1>

<div align="center">

このプロジェクトは、すべてを自動化することで、大規模な言語モデルを使用する際の障壁をなくすことを目的としています。必要なのは、
わずか数メガバイトの軽量な実行プログラムだけです。さらに、このプロジェクトは OpenAI API と互換性のあるインターフェイスを提供しており、
すべての ChatGPT クライアントは RWKV クライアントであることを意味します。

[![license][license-image]][license-url]
[![release][release-image]][release-url]
[![py-version][py-version-image]][py-version-url]

[English](README.md) | [简体中文](README_ZH.md) | 日本語

### インストール

[![Windows][Windows-image]][Windows-url]
[![MacOS][MacOS-image]][MacOS-url]
[![Linux][Linux-image]][Linux-url]

[FAQs](https://github.com/josStorer/RWKV-Runner/wiki/FAQs) | [プレビュー](#Preview) | [ダウンロード][download-url] | [シンプルなデプロイの例](#Simple-Deploy-Example) | [サーバーデプロイ例](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples) | [MIDIハードウェア入力](#MIDI-Input)

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

## ヒント

- サーバーに [backend-python](./backend-python/)
  をデプロイし、このプログラムをクライアントとして使用することができます。設定された`API URL`にサーバーアドレスを入力してください。

- もし、あなたがデプロイし、外部に公開するサービスを提供している場合、APIゲートウェイを使用してリクエストのサイズを制限し、
  長すぎるプロンプトの提出がリソースを占有しないようにしてください。さらに、実際の状況に応じて、リクエストの max_tokens
  の上限を制限してください：https://github.com/josStorer/RWKV-Runner/blob/master/backend-python/utils/rwkv.py#L567
  、デフォルトは le=102400 ですが、極端な場合には単一の応答が大量のリソースを消費する可能性があります。

- デフォルトの設定はカスタム CUDA カーネルアクセラレーションを有効にしています。互換性の問題 (文字化けを出力する)
  が発生する可能性がある場合は、コンフィグページに移動し、`Use Custom CUDA kernel to Accelerate`
  をオフにしてください、あるいは、GPUドライバーをアップグレードしてみてください。

- Windows Defender
  がこれをウイルスだと主張する場合は、[v1.3.7_win.zip](https://github.com/josStorer/RWKV-Runner/releases/download/v1.3.7/RWKV-Runner_win.zip)
  をダウンロードして最新版に自動更新させるか、信頼済みリストに追加してみてください (`Windows Security` -> `Virus & threat protection` -> `Manage settings` -> `Exclusions` -> `Add or remove exclusions` -> `Add an exclusion` -> `Folder` -> `RWKV-Runner`)。

- 異なるタスクについては、API パラメータを調整することで、より良い結果を得ることができます。例えば、翻訳タスクの場合、Temperature
  を 1 に、Top_P を 0.3 に設定してみてください。

## 特徴

- RWKV モデル管理とワンクリック起動
- フロントエンドとバックエンドの分離は、クライアントを使用しない場合でも、フロントエンドサービス、またはバックエンド推論サービス、またはWebUIを備えたバックエンド推論サービスを個別に展開することを可能にします。
  [シンプルなデプロイの例](#Simple-Deploy-Example) | [サーバーデプロイ例](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples)
- OpenAI API と互換性があり、すべての ChatGPT クライアントを RWKV クライアントにします。モデル起動後、
  http://127.0.0.1:8000/docs を開いて詳細をご覧ください。
- 依存関係の自動インストールにより、軽量な実行プログラムのみを必要とします
- 事前設定された多段階のVRAM設定、ほとんどのコンピュータで動作します。配置ページで、ストラテジーをWebGPUに切り替えると、AMD、インテル、その他のグラフィックカードでも動作します
- ユーザーフレンドリーなチャット、完成、および作曲インターフェイスが含まれています。また、チャットプリセット、添付ファイルのアップロード、MIDIハードウェア入力、トラック編集もサポートしています。
  [プレビュー](#Preview) | [MIDIハードウェア入力](#MIDI-Input)
- 内蔵WebUIオプション、Webサービスのワンクリック開始、ハードウェアリソースの共有
- 分かりやすく操作しやすいパラメータ設定、各種操作ガイダンスプロンプトとともに
- 内蔵モデル変換ツール
- ダウンロード管理とリモートモデル検査機能内蔵
- 内蔵のLoRA微調整機能を搭載しています (Windowsのみ)
- このプログラムは、OpenAI ChatGPTとGPT Playgroundのクライアントとしても使用できます（設定ページで `API URL` と `API Key`
  を入力してください）
- 多言語ローカライズ
- テーマ切り替え
- 自動アップデート

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

## API 同時実行ストレステスト

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

## 埋め込み API の例

注意: v1.4.0 では、埋め込み API の品質が向上しました。生成される結果は、以前のバージョンとは互換性がありません。
もし、embeddings API を使って知識ベースなどを生成している場合は、再生成してください。

LangChain を使用している場合は、`OpenAIEmbeddings(openai_api_base="http://127.0.0.1:8000", openai_api_key="sk-")`
を使用してください

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

MIDIキーボードをお持ちでない場合、`Virtual Midi Controller 3 LE`
などの仮想MIDI入力ソフトウェアを使用することができます。[loopMIDI](https://www.tobias-erichsen.de/wp-content/uploads/2020/01/loopMIDISetup_1_0_16_27.zip)
を組み合わせて、通常のコンピュータキーボードをMIDI入力として使用できます。

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

## 関連リポジトリ:

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

### ホームページ

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c9b9cdd0-63f9-4319-9f74-5bf5d7df5a67)

### チャット

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/80009872-528f-4932-aeb2-f724fa892e7c)

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/e98c9038-3323-47b0-8edb-d639fafd37b2)

### 補完

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/bf49de8e-3b89-4543-b1ef-7cd4b19a1836)

### 作曲

Tip: You can download https://github.com/josStorer/sgm_plus and unzip it to the program's `assets/sound-font` directory
to use it as an offline sound source. Please note that if you are compiling the program from source code, do not place
it in the source code directory.

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/e8ad908d-3fd2-4e92-bcdb-96815cb836ee)

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/b2ce4761-9e75-477e-a182-d0255fb8ac76)

### コンフィグ

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/f41060dc-5517-44af-bb3f-8ef71720016d)

### モデル管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/b1581147-a6ce-4493-8010-e33c0ddeca0a)

### ダウンロード管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c8153cf9-c8cb-4618-8268-60c82a5be539)

### LoRA Finetune

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/4715045a-683e-4d2a-9b0e-090c7a5df63f)

### 設定

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/1067e635-8c07-4217-86a8-e48a5fcbb075)
