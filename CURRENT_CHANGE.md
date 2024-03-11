## Changes

### Features

- add Docker support (#291) @LonghronShen

### Fixes

- fix a generation exception caused by potentially dangerous regex being passed into the stop array
- fix max_tokens parameter of Chat page not being passed to backend
- fix the issue where penalty_decay and global_penalty are not being passed to the backend default config when running
  the model through client

### Improvements

- prevent 'torch' has no attribute 'cuda' error in torch_gc, so user can use CPU or WebGPU (#302)

### Chores

- bump dependencies
- add pre-release workflow
- dep_check.py now ignores GPUtil

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
