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

[Preview](#Preview) | [Download][download-url]

[license-image]: http://img.shields.io/badge/license-MIT-blue.svg

[license-url]: https://github.com/josStorer/RWKV-Runner/blob/master/LICENSE

[release-image]: https://img.shields.io/github/release/josStorer/RWKV-Runner.svg

[release-url]: https://github.com/josStorer/RWKV-Runner/releases/latest

[download-url]: https://github.com/josStorer/RWKV-Runner/releases/download/v1.0.2/RWKV-Runner_windows_x64.exe

</div>

#### Default configs do not enable custom CUDA kernel acceleration, but I strongly recommend that you enable it and run with int8 precision, which is much faster and consumes much less VRAM. Go to the Configs page and turn on `Use Custom CUDA kernel to Accelerate`.

#### For different tasks, adjusting API parameters can achieve better results. For example, for translation tasks, you can try setting Temperature to 1 and Top_P to 0.3.

## Features

- RWKV model management and one-click startup
- Fully compatible with the OpenAI API, making every ChatGPT client an RWKV client. After starting the model,
  open http://127.0.0.1:8000/docs to view more details.
- Automatic dependency installation, requiring only a lightweight executable program
- User-friendly chat interaction interface included
- Easy-to-understand and operate parameter configuration
- Built-in model conversion tool
- Built-in download management and remote model inspection
- Multilingual localization
- Theme switching
- Automatic updates

## Todo

- [ ] Model training functionality
- [x] CUDA operator int8 acceleration
- [ ] macOS support
- [ ] Linux support

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
