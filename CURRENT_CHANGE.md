## Changes

### Upgrades

- web-rwkv-py 0.1.2 (Support V4, V5 and V6) https://github.com/cryscan/web-rwkv-py
- webgpu 0.3.13 https://github.com/cgisky1980/ai00_rwkv_server

### Features

- add markdown renderer switch
- allow loading conversation
- allow setting history message number
- expose penalty_decay, top_k
- add AVOID_PENALTY_TOKENS
- add [parse_api_log.py](https://github.com/josStorer/RWKV-Runner/blob/master/parse_api_log.py), this script can extract
  formatted data from api.log

### Improvements

- improve macos experience
- improve fine-tune performance
- add better custom tokenizer support and tokenizer-midipiano.json
- improve path processing
- add EOS state cache point
- reduce package size

### Fixes

- fix WSL2 WindowsOptionalFeature: Microsoft-Windows-Subsystem-Linux -> VirtualMachinePlatform
- fix finetune errorsMap ($modelInfo)

### Chores

- update defaultPresets
- update defaultModelConfigs
- rename manifest tag "Main" -> "Official"
- update manifest.json
- update Related Repositories
- other minor changes

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
