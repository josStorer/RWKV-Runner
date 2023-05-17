import {makeAutoObservable} from 'mobx';
import {saveConfigs} from '../utils';

export enum ModelStatus {
  Offline,
  Starting,
  Loading,
  Working,
}

export type ModelSourceItem = {
  name: string;
  size: number;
  lastUpdated: string;
  desc?: { [lang: string]: string; };
  SHA256?: string;
  url?: string;
  downloadUrl?: string;
  isLocal?: boolean;
  isDownloading?: boolean;
  lastUpdatedMs?: number;
};

export type ApiParameters = {
  apiPort: number
  maxResponseToken: number;
  temperature: number;
  topP: number;
  presencePenalty: number;
  frequencyPenalty: number;
}

export type Device = 'CPU' | 'CUDA';
export type Precision = 'fp16' | 'int8' | 'fp32';

export type ModelParameters = {
  // different models can not have the same name
  modelName: string;
  device: Device;
  precision: Precision;
  storedLayers: number;
  maxStoredLayers: number;
  enableHighPrecisionForLastLayer: boolean;
}

export type ModelConfig = {
  // different configs can have the same name
  name: string;
  apiParameters: ApiParameters
  modelParameters: ModelParameters
}

export const defaultModelConfigs: ModelConfig[] = [
  {
    name: 'Default',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 4100,
      temperature: 1,
      topP: 1,
      presencePenalty: 0,
      frequencyPenalty: 0
    },
    modelParameters: {
      modelName: 'RWKV-4-Raven-1B5-v11-Eng99%-Other1%-20230425-ctx4096.pth',
      device: 'CUDA',
      precision: 'fp16',
      storedLayers: 25,
      maxStoredLayers: 25,
      enableHighPrecisionForLastLayer: false
    }
  }
];

class CommonStore {
  constructor() {
    makeAutoObservable(this);
  }

  modelStatus: ModelStatus = ModelStatus.Offline;
  currentModelConfigIndex: number = 0;
  modelConfigs: ModelConfig[] = [];
  modelSourceManifestList: string = 'https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner/manifest.json;';
  modelSourceList: ModelSourceItem[] = [];

  getCurrentModelConfig = () => {
    return this.modelConfigs[this.currentModelConfigIndex];
  };

  setModelStatus = (status: ModelStatus) => {
    this.modelStatus = status;
  };

  setCurrentConfigIndex = (index: number, saveConfig: boolean = true) => {
    this.currentModelConfigIndex = index;
    if (saveConfig)
      saveConfigs();
  };

  setModelConfig = (index: number, config: ModelConfig, saveConfig: boolean = true) => {
    this.modelConfigs[index] = config;
    if (saveConfig)
      saveConfigs();
  };

  setModelConfigs = (configs: ModelConfig[], saveConfig: boolean = true) => {
    this.modelConfigs = configs;
    if (saveConfig)
      saveConfigs();
  };

  createModelConfig = (config: ModelConfig = defaultModelConfigs[0], saveConfig: boolean = true) => {
    if (config.name === defaultModelConfigs[0].name)
      config.name = new Date().toLocaleString();
    this.modelConfigs.push(config);
    if (saveConfig)
      saveConfigs();
  };

  deleteModelConfig = (index: number, saveConfig: boolean = true) => {
    this.modelConfigs.splice(index, 1);
    if (index < this.currentModelConfigIndex) {
      this.setCurrentConfigIndex(this.currentModelConfigIndex - 1);
    }
    if (this.modelConfigs.length === 0) {
      this.createModelConfig();
    }
    if (this.currentModelConfigIndex >= this.modelConfigs.length) {
      this.setCurrentConfigIndex(this.modelConfigs.length - 1);
    }
    if (saveConfig)
      saveConfigs();
  };

  setModelSourceManifestList = (value: string) => {
    this.modelSourceManifestList = value;
  };

  setModelSourceList = (value: ModelSourceItem[]) => {
    this.modelSourceList = value;
  };
}

export default new CommonStore();