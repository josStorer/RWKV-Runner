import { ModelConfig } from './Configs';

export const defaultModelConfigsMac: ModelConfig[] = [
  {
    name: 'MAC-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: false,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'MAC-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: false,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'MAC-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'MPS',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: false,
      customStrategy: 'mps fp32'
    }
  },
  {
    name: 'CPU-6G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-12G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-12G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-28G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-28G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  }
];

export const defaultModelConfigs: ModelConfig[] = [
  {
    name: 'GPU-2G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 4,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 8,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-4G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 8,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 18,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-6G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 18,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: true,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 27,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-8G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 27,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-10G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-10G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-12G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 24,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-16G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-16G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-16G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 37,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-18G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'int8',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'GPU-32G-14B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false,
      useCustomCuda: true
    }
  },
  {
    name: 'CPU-6G-1B5-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-12G-3B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng98%-Other2%-20230520-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-12G-3B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-28G-7B-EN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  },
  {
    name: 'CPU-28G-7B-CN',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1.2,
      topP: 0.5,
      presencePenalty: 0.4,
      frequencyPenalty: 0.4
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-7B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230530-ctx8192.pth',
      device: 'CPU',
      precision: 'fp32',
      storedLayers: 41,
      maxStoredLayers: 41,
      enableHighPrecisionForLastLayer: false
    }
  }
];