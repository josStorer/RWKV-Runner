## Changes

- add torch-2.7.1+cu128 precompiled kernels
![image](https://github.com/user-attachments/assets/aa1703ec-da0e-4e3f-820c-9253f4b9bf15)
- hide unnecessary pop-up consoles on windows
- The linux binary files released in github releases now depend on libwebkit2gtk-4.1 to support Ubuntu 24.04. This means that versions below Ubuntu 20.04 will no longer be supported for running, and users will have to build it on their own. Additionally, Windows 7 is still supported, but you need to install the KB2999226 patch.
- add quick think support
![Image](https://github.com/user-attachments/assets/ecf66622-0765-42c9-b8a0-633c30329349)
- fix the issue where the line breaks in the thinking content did not take effect
- update manifest.json and defaultModelConfigs
- bump go-webview2

Note: If you encounter WebView2 crash issues, please try opening the Windows Settings, click on Apps, search for
WebView2, click Modify -> Repair to update your WebView2 runtime.

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
