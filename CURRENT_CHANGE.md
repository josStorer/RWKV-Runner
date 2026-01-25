## Changes

- disable auto throttling of @microsoft/fetch-event-source
- bump precompiled llama.cpp vulkan
- add `stop_token_ids` field to `/completions` and `/chat/completions`
- allow stop sequences to accept multiple values or numeric token IDs, passed to the api server as arrays
  <img width="331" height="170" alt="Image" src="https://github.com/user-attachments/assets/3dcd7a8b-9503-4fc4-8f89-6b0b8e33e933" />
- update manifest.json and defaultConfigs
- small fixes

Note: If you encounter WebView2 crash issues, please try opening the Windows Settings, click on Apps, search for
WebView2, click Modify -> Repair to update your WebView2 runtime.

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
- Windows 7 Patches: https://github.com/josStorer/wails/releases/tag/v2.9.2x
