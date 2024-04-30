## Changes

- bump webgpu mode [ai00_server v0.4.2](https://github.com/Ai00-X/ai00_server) (huge performance improvement)
- upgrade to rwkv 0.8.26 (state-tuned model support)
- update defaultConfigs and manifest.json
- chores

## Breaking Changes

- change the default value of `presystem` to false

For the convenience of using the future state-tuned models, the default value of `presystem` has been set to false. This
means that the RWKV-Runner service will no longer automatically insert recommended RWKV pre-prompts for you:

```
User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.
```

If you are using the API service and conducting a rigorous RWKV conversation, please manually send the above messages to
the `/chat/completions` API's `messages` array, or manually send `presystem: true` to have the server automatically
insert pre-prompts.

If you are using the RWKV-Runner client for chatting, you can enable `Insert default system prompt at the beginning` in
the preset editor.

Of course, in reality, even if you do not perform the above, there is usually no significant negative impact.

If you are using the new RWKV state-tuned models, you do not need to perform the above.

The new RWKV state-tuned models can be downloaded here, they are very interesting:

- https://huggingface.co/BlinkDL/rwkv-6-state-instruct-aligned
- https://huggingface.co/BlinkDL/temp-latest-training-models

If you are interested in state-tuning, please refer
to: https://github.com/BlinkDL/RWKV-LM#state-tuning-tuning-the-initial-state-zero-inference-overhead

## Install

- Windows: https://github.com/josStorer/RWKV-Runner/blob/master/build/windows/Readme_Install.txt
- MacOS: https://github.com/josStorer/RWKV-Runner/blob/master/build/darwin/Readme_Install.txt
- Linux: https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt
- Simple Deploy Example: https://github.com/josStorer/RWKV-Runner/blob/master/README.md#simple-deploy-example
- Server Deploy Examples: https://github.com/josStorer/RWKV-Runner/tree/master/deploy-examples
