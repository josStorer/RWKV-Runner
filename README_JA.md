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

[English](README.md) | [简体中文](README_ZH.md) | 日本語

### インストール

[![Windows][Windows-image]][Windows-url]
[![MacOS][MacOS-image]][MacOS-url]
[![Linux][Linux-image]][Linux-url]

[FAQs](https://github.com/josStorer/RWKV-Runner/wiki/FAQs) | [プレビュー](#Preview) | [ダウンロード][download-url] | [サーバーデプロイ例](https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples)

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

#### デフォルトの設定はカスタム CUDA カーネルアクセラレーションを有効にしています。互換性の問題が発生する可能性がある場合は、コンフィグページに移動し、`Use Custom CUDA kernel to Accelerate` をオフにしてください。

#### Windows Defender がこれをウイルスだと主張する場合は、[v1.0.8](https://github.com/josStorer/RWKV-Runner/releases/tag/v1.0.8) / [v1.0.9](https://github.com/josStorer/RWKV-Runner/releases/tag/v1.0.9) をダウンロードして最新版に自動更新させるか、信頼済みリストに追加してみてください。

#### 異なるタスクについては、API パラメータを調整することで、より良い結果を得ることができます。例えば、翻訳タスクの場合、Temperature を 1 に、Top_P を 0.3 に設定してみてください。

## 特徴

- RWKV モデル管理とワンクリック起動
- OpenAI API と完全に互換性があり、すべての ChatGPT クライアントを RWKV クライアントにします。モデル起動後、
  http://127.0.0.1:8000/docs を開いて詳細をご覧ください。
- 依存関係の自動インストールにより、軽量な実行プログラムのみを必要とします
- 2G から 32G の VRAM のコンフィグが含まれており、ほとんどのコンピュータで動作します
- ユーザーフレンドリーなチャットと完成インタラクションインターフェースを搭載
- 分かりやすく操作しやすいパラメータ設定
- 内蔵モデル変換ツール
- ダウンロード管理とリモートモデル検査機能内蔵
- 内蔵のLoRA微調整機能を搭載しています
- このプログラムは、OpenAI ChatGPTとGPT Playgroundのクライアントとしても使用できます
- 多言語ローカライズ
- テーマ切り替え
- 自動アップデート

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

LangChain を使用している場合は、`OpenAIEmbeddings(openai_api_base="http://127.0.0.1:8000", openai_api_key="sk-")`を使用してください

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

## 関連リポジトリ:

- RWKV-4-World: https://huggingface.co/BlinkDL/rwkv-4-world/tree/main
- RWKV-4-Raven: https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main
- ChatRWKV: https://github.com/BlinkDL/ChatRWKV
- RWKV-LM: https://github.com/BlinkDL/RWKV-LM
- RWKV-LM-LoRA: https://github.com/Blealtan/RWKV-LM-LoRA

## プレビュー

### ホームページ

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/d7f24d80-f382-428d-8b28-edf87e1549e2)

### チャット

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/80009872-528f-4932-aeb2-f724fa892e7c)

### 補完

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/bf49de8e-3b89-4543-b1ef-7cd4b19a1836)

### コンフィグ

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/48befdc6-e03c-4851-9bee-22f77ee2640e)

### モデル管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/367fe4f8-cc12-475f-9371-3cf62cdbf293)

### ダウンロード管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/c8153cf9-c8cb-4618-8268-60c82a5be539)

### LoRA Finetune

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/4715045a-683e-4d2a-9b0e-090c7a5df63f)

### 設定

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/1067e635-8c07-4217-86a8-e48a5fcbb075)
