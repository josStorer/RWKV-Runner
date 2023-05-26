<p align="center">
    <img src="https://github.com/josStorer/RWKV-Runner/assets/13366013/d24834b0-265d-45f5-93c0-fac1e19562af">
</p>

<h1 align="center">RWKV Runner</h1>

<div align="center">

本项目旨在消除大语言模型的使用门槛，全自动为你处理一切，你只需要一个仅仅几MB的可执行程序。此外本项目提供了与OpenAI
API兼容的接口，这意味着一切ChatGPT客户端都是RWKV客户端。

[![license][license-image]][license-url]
[![release][release-image]][release-url]

[English](README.md) | 简体中文

[视频演示](https://www.bilibili.com/video/BV1hM4y1v76R) | [疑难解答](https://www.bilibili.com/read/cv23921171) | [预览](#Preview) | [下载][download-url]

[license-image]: http://img.shields.io/badge/license-MIT-blue.svg

[license-url]: https://github.com/josStorer/RWKV-Runner/blob/master/LICENSE

[release-image]: https://img.shields.io/github/release/josStorer/RWKV-Runner.svg

[release-url]: https://github.com/josStorer/RWKV-Runner/releases/latest

[download-url]: https://github.com/josStorer/RWKV-Runner/releases

</div>

#### 注意 目前RWKV中文模型质量一般，推荐使用英文模型体验实际RWKV能力

#### 预设配置没有开启自定义CUDA算子加速，但我强烈建议你开启它并使用int8量化运行，速度非常快，且显存消耗少得多。前往配置页面，打开`使用自定义CUDA算子加速`

#### 对于不同的任务，调整API参数会获得更好的效果，例如对于翻译任务，你可以尝试设置Temperature为1，Top_P为0.3

## 功能

- RWKV模型管理，一键启动
- 与OpenAI API完全兼容，一切ChatGPT客户端，都是RWKV客户端。启动模型后，打开 http://127.0.0.1:8000/docs 查看详细内容
- 全自动依赖安装，你只需要一个轻巧的可执行程序
- 自带用户友好的聊天交互页面
- 易于理解和操作的参数配置
- 内置模型转换工具
- 内置下载管理和远程模型检视
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

## Todo

- [ ] 模型训练功能
- [x] CUDA算子int8提速
- [ ] macOS支持
- [ ] linux支持

## 相关仓库:

- RWKV-4-Raven: https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main
- ChatRWKV: https://github.com/BlinkDL/ChatRWKV
- RWKV-LM: https://github.com/BlinkDL/RWKV-LM

## Preview

### 主页

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/9d25380a-a17b-443f-b823-86c754ebebf0)

### 聊天

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/0e66d5fa-f34a-409f-9cd4-d880815733f3)

### 补全

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/d4178ee9-a188-4878-9777-25c916872c29)

### 配置

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/ad9921fc-7248-40a3-9e18-03445b86e4bf)

### 模型管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/7c36f15f-3e77-49cd-a16d-99a29f870bdf)

### 下载管理

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/32fde30b-11dd-43b9-9667-ad6975be2106)

### 设置

![image](https://github.com/josStorer/RWKV-Runner/assets/13366013/e8a0f746-9da7-48e3-b3fc-e1453ac50de2)
