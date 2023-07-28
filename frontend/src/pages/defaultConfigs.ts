import { ModelConfig } from './Configs';
import { CompletionPreset } from './Completion';

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
    name: 'MAC-0.1B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-0.1B-v1-20230520-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'MAC-0.4B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-0.4B-v1-20230529-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32'
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
      modelName: 'RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'MAC-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
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
      modelName: 'RWKV-4-World-3B-v1-20230619-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'MAC-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
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
  },
  {
    name: 'CPU-120M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
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
    name: 'CPU-6G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-6G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-12G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-3B-v1-20230619-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-12G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-12G-3B-CN',
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
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-28G-7B-World',
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
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-28G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-28G-7B-CN',
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
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  }
];

export const defaultModelConfigs: ModelConfig[] = [
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
      modelName: 'RWKV-4-World-3B-v1-20230619-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 6,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-2G-0.1B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-0.1B-v1-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp32', // using fp16 will disable state cache (->)
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'GPU-2G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 4,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-0.4B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-0.4B-v1-20230529-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
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
      modelName: 'RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp32',
      storedLayers: 8,
      maxStoredLayers: 41
    }
  },
  {
    name: 'GPU-4G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
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
      modelName: 'RWKV-4-World-3B-v1-20230619-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
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
    name: 'GPU-4G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
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
    name: 'GPU-6G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
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
      modelName: 'RWKV-4-World-3B-v1-20230619-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
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
    name: 'GPU-6G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
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
    name: 'GPU-8G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
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
      modelName: 'RWKV-4-World-3B-v1-20230619-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
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
    name: 'GPU-8G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
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
    name: 'GPU-10G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
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
    name: 'GPU-12G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
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
    name: 'GPU-16G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
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
    name: 'GPU-16G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 37,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-18G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-32G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
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
      topP: 0.3,
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
    name: 'CPU-6G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-6G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-12G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-World-3B-v1-20230619-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-12G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-12G-3B-CN',
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
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-28G-7B-World',
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
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-28G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  },
  {
    name: 'CPU-28G-7B-CN',
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
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41
    }
  }
];