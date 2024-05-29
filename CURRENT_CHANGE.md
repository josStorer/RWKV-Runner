## v1.8.4

- fix f05a4a, __init__.py is not embedded

## v1.8.3

### Deprecations

- rwkv-beta is deprecated

### Upgrades

- bump webgpu(python) (https://github.com/cryscan/web-rwkv-py)
- sync https://github.com/JL-er/RWKV-PEFT (LoRA)

### Improvements

- improve default LoRA fine-tune params

### Fixes

- fix #342, #345: cannot import name 'packaging' from 'pkg_resources'
- fix the huge error prompt that pops up when running in webgpu mode

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
