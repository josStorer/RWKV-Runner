## Changes

### Llama.cpp support

- Add llama.cpp support with pre-compiled Vulkan libraries for Windows that should work out-of-the-box with any modern GPU. Mac and Linux users still need to manually install llama-cpp-python. You can now use RWKV GGUF models as well as any other GGUF models such as DeepSeek, Qwen3, Gemma3, Phi4. You can select the llama.cpp tag in the Models page and download the required models with one click, or place downloaded GGUF models in the models directory for use.
- <img src="https://github.com/user-attachments/assets/7ff56440-41cd-4ef2-8a63-b057061ddbf4" width="512"/>
- The software's preset configs have been streamlined and now include some GGUF format presets. You can click the reset button to fetch the latest presets.
- <img src="https://github.com/user-attachments/assets/b3203434-ccea-4043-8384-6cf412c37eb7" width="256"/>
- The RWKV-Runner Python server in llama.cpp mode has been optimized. After loading the model to GPU, the server process only occupies approximately 200MB of memory on Windows platform.
- When server users call the `/switch-model` API to load models, you only need to pass a file path ending with .gguf to the `model` field to use llama.cpp mode.

### Features

- llama.cpp support
- <img src="https://github.com/user-attachments/assets/50de157f-16cb-4d86-bb25-a1f13e2a031f" width="512"/>
- Add a setting to save the full rwkv-runner client state, rather than just storing necessary settings. This option is enabled by default. You can disable it and restart the software to restore the previous version's behavior
- <img src="https://github.com/user-attachments/assets/4c8c7b13-8d62-4ff0-ab0d-d06c40d95f58" width="512"/>
- add a share button to save your chat screenshot
- <img src="https://github.com/user-attachments/assets/11533e20-836c-487f-b3ae-5cf7988f12bc" width="512"/>

### Improvements
- reduce peak memory usage when loading rwkv7 in cuda mode
- increase the maximum value of the top_k API parameter to 100
- remove language tags in Models page, as all new models support global languages
- remove useless/disabled resources
- other small improvements
- You can run RWKV-Runner on Windows 7 by installing the patches from the link below. Note that you still need to install Python 3.8 and dependencies manually. https://github.com/josStorer/wails/releases/tag/v2.9.2x

### Fixes
- fix the issue of failing to load the state for RWKV7
- Fix the abnormal behavior when passing a Tool Definition array. This is a frontend only parameter construction issue.
- fix the issue where the model list did not refresh automatically after downloading the model when using a custom model path

Note: If you encounter WebView2 crash issues, please try opening the Windows Settings, click on Apps, search for
WebView2, click Modify -> Repair to update your WebView2 runtime.

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
- Windows 7 Patches: https://github.com/josStorer/wails/releases/tag/v2.9.2x
