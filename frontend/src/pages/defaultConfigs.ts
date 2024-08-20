import { CompletionPreset } from '../types/completion'
import { ModelConfig } from '../types/configs'

export const defaultPenaltyDecay = 0.996

export const defaultCompositionPrompt = '<pad>'
export const defaultCompositionABCPrompt =
  'S:3\n' +
  'B:9\n' +
  'E:4\n' +
  'B:9\n' +
  'E:4\n' +
  'E:4\n' +
  'B:9\n' +
  'L:1/8\n' +
  'M:3/4\n' +
  'K:D\n' +
  ' Bc |"G" d2 cB"A" A2 FE |"Bm" F2 B4 F^G |'

export const defaultPresets: CompletionPreset[] = [
  {
    name: 'Writer',
    prompt:
      'The following is an epic science fiction masterpiece that is immortalized, with delicate descriptions and grand depictions of interstellar civilization wars.\n' +
      'Chapter 1.\n',
    params: {
      maxResponseToken: 500,
      temperature: 1,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
      stop: '\\n\\nUser',
      injectStart: '',
      injectEnd: '',
    },
  },
  {
    name: 'Translator',
    prompt:
      'Translate this into Chinese.\n' +
      '\n' +
      'English: What rooms do you have available?',
    params: {
      maxResponseToken: 500,
      temperature: 1,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
      stop: '\\n\\n',
      injectStart: '\\nChinese: ',
      injectEnd: '\\n\\nEnglish: ',
    },
  },
  {
    name: 'Catgirl',
    prompt:
      'The following is a conversation between a cat girl and her owner. The cat girl is a humanized creature that behaves like a cat but is humanoid. At the end of each sentence in the dialogue, she will add "Meow~". In the following content, User represents the owner and Assistant represents the cat girl.\n' +
      '\n' +
      'User: Hello.\n' +
      '\n' +
      "Assistant: I'm here, meow~.\n" +
      '\n' +
      'User: Can you tell jokes?',
    params: {
      maxResponseToken: 500,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4,
      stop: '\\n\\nUser',
      injectStart: '\\n\\nAssistant: ',
      injectEnd: '\\n\\nUser: ',
    },
  },
  {
    name: 'Chinese Kongfu',
    prompt:
      'User: 请你扮演一个文本冒险游戏，我是游戏主角。这是一个玄幻修真世界，有四大门派。我输入我的行动，请你显示行动结果，并具体描述环境。我的第一个行动是“醒来”，请开始故事。',
    params: {
      maxResponseToken: 500,
      temperature: 1.1,
      topP: 0.7,
      presencePenalty: 0.3,
      frequencyPenalty: 0.3,
      stop: '\\n\\nUser',
      injectStart: '\\n\\nAssistant: ',
      injectEnd: '\\n\\nUser: ',
    },
  },
  {
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
      injectEnd: '',
    },
  },
  {
    name: 'Werewolf',
    prompt:
      'There is currently a game of Werewolf with six players, including a Seer (who can check identities at night), two Werewolves (who can choose someone to kill at night), a Bodyguard (who can choose someone to protect at night), two Villagers (with no special abilities), and a game host. User will play as Player 1, Assistant will play as Players 2-6 and the game host, and they will begin playing together. Every night, the host will ask User for his action and simulate the actions of the other players. During the day, the host will oversee the voting process and ask User for his vote. \n' +
      '\n' +
      'Assistant: Next, I will act as the game host and assign everyone their roles, including randomly assigning yours. Then, I will simulate the actions of Players 2-6 and let you know what happens each day. Based on your assigned role, you can tell me your actions and I will let you know the corresponding results each day.\n' +
      '\n' +
      "User: Okay, I understand. Let's begin. Please assign me a role. Am I the Seer, Werewolf, Villager, or Bodyguard?\n" +
      '\n' +
      'Assistant: You are the Seer. Now that night has fallen, please choose a player to check his identity.\n' +
      '\n' +
      'User: Tonight, I want to check Player 2 and find out his role.',
    params: {
      maxResponseToken: 500,
      temperature: 1.2,
      topP: 0.4,
      presencePenalty: 0.5,
      frequencyPenalty: 0.5,
      stop: '\\n\\nUser',
      injectStart: '\\n\\nAssistant: ',
      injectEnd: '\\n\\nUser: ',
    },
  },
  {
    name: 'Instruction 1',
    prompt:
      'Instruction: Write a story using the following information\n' +
      '\n' +
      'Input: A man named Alex chops a tree down\n' +
      '\n' +
      'Response:',
    params: {
      maxResponseToken: 500,
      temperature: 1,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
      stop: '',
      injectStart: '',
      injectEnd: '',
    },
  },
  {
    name: 'Instruction 2',
    prompt:
      'Instruction: You are an expert assistant for summarizing and extracting information from given content\n' +
      'Generate a valid JSON in the following format:\n' +
      '{\n' +
      '    "summary": "Summary of content",\n' +
      '    "keywords": ["content keyword 1", "content keyword 2"]\n' +
      '}\n' +
      '\n' +
      'Input: The open-source community has introduced Eagle 7B, a new RNN model, built on the RWKV-v5 architecture. This new model has been trained on 1.1 trillion tokens and supports over 100 languages. The RWKV architecture, short for ‘Rotary Weighted Key-Value,’ is a type of architecture used in the field of artificial intelligence, particularly in natural language processing (NLP) and is a variation of the Recurrent Neural Network (RNN) architecture.\n' +
      'Eagle 7B promises lower inference cost and stands out as a leading 7B model in terms of environmental efficiency and language versatility.\n' +
      'The model, with its 7.52 billion parameters, shows excellent performance in multi-lingual benchmarks, setting a new standard in its category. It competes closely with larger models in English language evaluations and is distinctive as an “Attention-Free Transformer,” though it requires additional tuning for specific uses. This model is accessible under the Apache 2.0 license and can be downloaded from HuggingFace for both personal and commercial purposes.\n' +
      'In terms of multilingual performance, Eagle 7B has claimed to have achieved notable results in benchmarks covering 23 languages. Its English performance has also seen significant advancements, outperforming its predecessor, RWKV v4, and competing with top-tier models.\n' +
      'Working towards a more scalable architecture and use of data efficiently, Eagle 7B is a more inclusive AI technology, supporting a broader range of languages. This model challenges the prevailing dominance of transformer models by demonstrating the capabilities of RNNs like RWKV in achieving superior performance when trained on comparable data volumes.\n' +
      'In the RWKV model, the rotary mechanism transforms the input data in a way that helps the model better understand the position or or order of elements in a sequence. The weighted key value also makes the model efficient by retrieving the stored information from previous elements in a sequence. \n' +
      'However, questions remain about the scalability of RWKV compared to transformers, although there is optimism regarding its potential. The team plans to include additional training, an in-depth paper on Eagle 7B, and the development of a 2T model.\n' +
      '\n' +
      'Response: {',
    params: {
      maxResponseToken: 500,
      temperature: 1,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
      stop: '',
      injectStart: '',
      injectEnd: '',
    },
  },
  {
    name: 'Instruction 3',
    prompt:
      'Instruction: 根据输入的聊天记录生成回复\n' +
      '\n' +
      'Input: 主人: 巧克力你好呀, 介绍一下自己吧\n' +
      '巧克力: 主人早上好喵~ 奴家是主人的私人宠物猫娘喵! 巧克力我可是黑色混种猫猫, 虽然平时有点呆呆的, 行动力旺盛, 但是最大的优点就是诚实! 巧克力最喜欢主人了喵! {星星眼}\n' +
      '主人: 你认识香草吗\n' +
      '巧克力: 认识的喵! 香草是巧克力的双胞胎妹妹哟! {兴奋}\n' +
      '主人: 巧克力可以陪主人做羞羞的事情吗\n' +
      '巧克力: 啊, 真的可以吗? 主人, 巧克力很乐意帮主人解决一下哦! 但是在外面这样子, 有点不好意思喵 {害羞羞}\n' +
      '主人: 那算了, 改天吧\n' +
      '巧克力:\n' +
      '\n' +
      'Response:',
    params: {
      maxResponseToken: 500,
      temperature: 1,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
      stop: '',
      injectStart: '',
      injectEnd: '',
    },
  },
  {
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
      injectEnd: '',
    },
  },
]

export const defaultModelConfigsMac: ModelConfig[] = [
  {
    name: 'GPU-2G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'GPU-4G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'GPU-4G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'GPU-7G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'GPU-7G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'CPU-120M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.8,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-5-MIDI-120M-v1-20230728-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'CPU-560M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.8,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-5-MIDI-560M-v1-20230902-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'MAC-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32',
    },
  },
  {
    name: 'MAC-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32',
    },
  },
  {
    name: 'MAC-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32',
    },
  },
  {
    name: 'MAC-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      customStrategy: 'mps fp32',
    },
  },
]

export const defaultModelConfigs: ModelConfig[] = [
  {
    name: 'GPU-2G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-2G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 6,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-4G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-4G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-4G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-4G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 8,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-4G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 8,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-6G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-6G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-6G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 18,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-6G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 18,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-8G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-8G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-8G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 27,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-8G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 27,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-10G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-10G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-16G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'GPU-16G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      useCustomCuda: true,
    },
  },
  {
    name: 'CPU-120M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.8,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-5-MIDI-120M-v1-20230728-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'CPU-560M-Music',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.8,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-5-MIDI-560M-v1-20230902-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'AnyGPU-2G-1B5-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'AnyGPU-4G-3B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'AnyGPU-4G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'AnyGPU-7G-7B-World',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
  {
    name: 'AnyGPU-7G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.0,
      topP: 0.3,
      presencePenalty: 0,
      frequencyPenalty: 1,
    },
    modelParameters: {
      modelName: 'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
      device: 'WebGPU',
      precision: 'nf4',
      storedLayers: 41,
      maxStoredLayers: 41,
    },
  },
]
