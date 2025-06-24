## Changes

- Add NVIDIA hardware info display to Settings Page with PyTorch version switching capability. Auto-select optimal PyTorch version during initial setup based on detected hardware. (Currently only works on Windows)
![image](https://github.com/user-attachments/assets/cce4b8ce-a920-451d-8f5f-c497b06a6339)
![image](https://github.com/user-attachments/assets/aa1703ec-da0e-4e3f-820c-9253f4b9bf15)
- temporarily disable the standard WebGPU strategy as it's outdated
- improve details

Note: If you encounter WebView2 crash issues, please try opening the Windows Settings, click on Apps, search for
WebView2, click Modify -> Repair to update your WebView2 runtime.

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
