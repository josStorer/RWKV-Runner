## Changes

- bump rwkv pip (improve VRAM usage when using rwkv7)
- the reasoning model renderer no longer modifies the original response's `<think>` tags, but only processes them during the rendering process, and fixes the issue where markdown was not correctly rendered when rendering the `<think>` tags in certain cases
- update the shortcut API list and model list in the settings, add OpenRouter and DeepSeek, and update the list with the most commonly used models at present
- update manifest (add rwkv7-g1 reasoning model)
- add `make devq` command to improve the startup and reload speed during project development. Requires `go install github.com/josStorer/wails/v2/cmd/wails@v2.9.2x`

Note: If you encounter WebView2 crash issues, please try opening the Windows Settings, click on Apps, search for
WebView2, click Modify -> Repair to update your WebView2 runtime.

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
