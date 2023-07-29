## Breaking Changes

Due to performance and bandwidth considerations, the `/chat/completions` and `/completions` API no longer return
the `response` field. If necessary, you can
still [uncomment it](https://github.com/josStorer/RWKV-Runner/commit/aecacde81927e26816558f1a629cdcf507b7cb5b) yourself.
Please note that this is never part of the OpenAI API, it existed previously only for API development
convenience. If you follow the OpenAI API specification, you will not be affected in any way.

## Changes

- improve `/chat/completions` and `/completions` API performance (remove `response` field)
- improve default ChatCompletion `stop`
- improve python backend startup speed
- update defaultConfigs

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Server-Deploy-Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples