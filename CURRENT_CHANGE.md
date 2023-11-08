## Changes

### Features

- add webUI for easier service sharing (enable it in Configs page or --webui command line parameter, compile it
  with `make
  build-web`)
- add deployment mode. If `/switch-model` with `deploy: true`, will disable /switch-model, /exit and other dangerous
  APIs (state cache APIs, part of midi APIs)

### Chores

- print error.txt to console when script fails
- api url getter

### Fixes

- set deepspeed to 0.11.2 to avoid finetune error
- fix `/docs` default api params (Pydantic v2)

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Server-Deploy-Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples