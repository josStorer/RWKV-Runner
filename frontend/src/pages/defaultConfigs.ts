import { CompletionPreset } from '../types/completion';
import { ModelConfig } from '../types/configs';

export const defaultCompositionPrompt = '<pad>';

export const defaultPresets: CompletionPreset[] = [{
  name: 'Writer',
  prompt: 'The following is an epic science fiction masterpiece that is immortalized, with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n',
  params: {
    maxResponseToken: 500,
    temperature: 1.2,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: '\\n\\nUser',
    injectStart: '',
    injectEnd: ''
  }
}, {
  name: 'Translator',
  prompt: 'Translate this into Chinese.\n\nEnglish: What rooms do you have available?',
  params: {
    maxResponseToken: 500,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0,
    frequencyPenalty: 1,
    stop: '\\n\\n',
    injectStart: '\\nChinese: ',
    injectEnd: '\\n\\nEnglish: '
  }
}, {
  name: 'Catgirl',
  prompt: 'The following is a conversation between a cat girl and her owner. The cat girl is a humanized creature that behaves like a cat but is humanoid. At the end of each sentence in the dialogue, she will add \"Meow~\". In the following content, User represents the owner and Assistant represents the cat girl.\n\nUser: Hello.\n\nAssistant: I\'m here, meow~.\n\nUser: Can you tell jokes?',
  params: {
    maxResponseToken: 500,
    temperature: 1.2,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: '\\n\\nUser',
    injectStart: '\\n\\nAssistant: ',
    injectEnd: '\\n\\nUser: '
  }
}, {
  name: 'Chinese Kongfu',
  prompt: 'User: 请你扮演一个文本冒险游戏，我是游戏主角。这是一个玄幻修真世界，有四大门派。我输入我的行动，请你显示行动结果，并具体描述环境。我的第一个行动是“醒来”，请开始故事。',
  params: {
    maxResponseToken: 500,
    temperature: 1.1,
    topP: 0.7,
    presencePenalty: 0.3,
    frequencyPenalty: 0.3,
    stop: '\\n\\nUser',
    injectStart: '\\n\\nAssistant: ',
    injectEnd: '\\n\\nUser: '
  }
}, {
  name: 'Code Generation',
  prompt: 'def sum(',
  params: {
    maxResponseToken: 500,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0,
    frequencyPenalty: 1,
    stop: '\\n\\n',
    injectStart: '',
    injectEnd: ''
  }
}, {
  name: 'Werewolf',
  prompt: 'There is currently a game of Werewolf with six players, including a Seer (who can check identities at night), two Werewolves (who can choose someone to kill at night), a Bodyguard (who can choose someone to protect at night), two Villagers (with no special abilities), and a game host. User will play as Player 1, Assistant will play as Players 2-6 and the game host, and they will begin playing together. Every night, the host will ask User for his action and simulate the actions of the other players. During the day, the host will oversee the voting process and ask User for his vote. \n\nAssistant: Next, I will act as the game host and assign everyone their roles, including randomly assigning yours. Then, I will simulate the actions of Players 2-6 and let you know what happens each day. Based on your assigned role, you can tell me your actions and I will let you know the corresponding results each day.\n\nUser: Okay, I understand. Let\'s begin. Please assign me a role. Am I the Seer, Werewolf, Villager, or Bodyguard?\n\nAssistant: You are the Seer. Now that night has fallen, please choose a player to check his identity.\n\nUser: Tonight, I want to check Player 2 and find out his role.',
  params: {
    maxResponseToken: 500,
    temperature: 1.2,
    topP: 0.4,
    presencePenalty: 0.5,
    frequencyPenalty: 0.5,
    stop: '\\n\\nUser',
    injectStart: '\\n\\nAssistant: ',
    injectEnd: '\\n\\nUser: '
  }
}, {
  name: 'Instruction',
  prompt: 'Instruction: Write a story using the following information\n\nInput: A man named Alex chops a tree down\n\nResponse:',
  params: {
    maxResponseToken: 500,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0,
    frequencyPenalty: 1,
    stop: '',
    injectStart: '',
    injectEnd: ''
  }
}, {
  name: 'Blank',
  prompt: '',
  params: {
    maxResponseToken: 500,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0,
    frequencyPenalty: 1,
    stop: '',
    injectStart: '',
    injectEnd: ''
  }
}];

export const defaultModelConfigsMac: ModelConfig[] = [
  {
    name: 'GPU-2G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'GPU-4G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-3B-v2-20231118-ctx16k.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'GPU-4G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'GPU-7G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-7B-v1-20230626-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'GPU-7G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-120M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.8,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-MIDI-120M-v1-20230714-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-560M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.8,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-MIDI-560M-v1-20230717-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'MAC-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'MAC-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-3B-v2-20231118-ctx16k.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'MAC-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'MAC-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-7B-v1-20230626-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32'
    }
  }
];

export const defaultModelConfigs: ModelConfig[] = [
  {
    name: 'GPU-2G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-2G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-3B-v2-20231118-ctx16k.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 6,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-3B-v2-20231118-ctx16k.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-7B-v1-20230626-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 8,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 8,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-3B-v2-20231118-ctx16k.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-7B-v1-20230626-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 18,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 18,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-3B-v2-20231118-ctx16k.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-7B-v1-20230626-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 27,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 27,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-10G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-7B-v1-20230626-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-10G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-16G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-7B-v1-20230626-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-16G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'CPU-120M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.8,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-MIDI-120M-v1-20230714-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-560M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.8,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-MIDI-560M-v1-20230717-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'AnyGPU-2G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'AnyGPU-4G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-5-World-3B-v2-20231118-ctx16k.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'AnyGPU-4G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-3B-v1-20230625-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'AnyGPU-7G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-7B-v1-20230626-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'AnyGPU-7G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  }
];