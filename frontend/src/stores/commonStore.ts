import {makeAutoObservable} from 'mobx';

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
  countPenalty: number;
}

export type ModelParameters = {
  modelName: string;
  device: string;
  precision: string;
  streamedLayers: number;
  enableHighPrecisionForLastLayer: boolean;
}

export type ModelConfig = {
  configName: string;
  apiParameters: ApiParameters
  modelParameters: ModelParameters
}

const defaultModelConfigs: ModelConfig[] = [
  {
    configName: 'Default',
    apiParameters: {
      apiPort: 8000,
      maxResponseToken: 1000,
      temperature: 1,
      topP: 1,
      presencePenalty: 0,
      countPenalty: 0
    },
    modelParameters: {
      modelName: '124M',
      device: 'CPU',
      precision: 'fp32',
      streamedLayers: 1,
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
  modelConfigs: ModelConfig[] = defaultModelConfigs;
  modelSourceManifestList: string = 'https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner/manifest.json;';
  modelSourceList: ModelSourceItem[] = [];

  setModelStatus = (status: ModelStatus) => {
    this.modelStatus = status;
  };

  setCurrentConfigIndex = (index: number) => {
    this.currentModelConfigIndex = index;
  };

  setModelConfig = (index: number, config: ModelConfig) => {
    this.modelConfigs[index] = config;
  };

  createModelConfig = (config: ModelConfig = defaultModelConfigs[0]) => {
    this.modelConfigs.push(config);
  };

  deleteModelConfig = (index: number) => {
    this.modelConfigs.splice(index, 1);
  };

  setModelSourceManifestList = (value: string) => {
    this.modelSourceManifestList = value;
  };

  setModelSourceList = (value: ModelSourceItem[]) => {
    this.modelSourceList = value;
  };
}

export default new CommonStore();