## Changes

**This version includes important bug fixes, it is strongly recommended to upgrade to this version.**

### Upgrades

- webgpu 0.3.20 https://github.com/cgisky1980/ai00_rwkv_server

### Features

- allow setting quantizedLayers of WebGPU mode

### Improvements

- improve occurrence[token] condition
- disable AVOID_PENALTY_TOKENS when generating (still enabled when preprocessing)
- enable useHfMirror by default for chinese users

### Fixes

- fix the issue where state cache could be modified leading to inconsistent hit results
- fix convert_safetensors.py for rwkv6
- add python3-dev to lora fine-tune dependencies (this may previously lead to the error of v5 fine-tune)

### Chores

- hide MPS and CUDA-Beta Options
- update manifest

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
