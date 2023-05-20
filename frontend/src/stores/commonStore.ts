import {makeAutoObservable} from 'mobx';
import {getUserLanguage, isSystemLightMode, saveConfigs} from '../utils';
import {WindowSetDarkTheme, WindowSetLightTheme} from '../../wailsjs/runtime';
import manifest from '../../../manifest.json';
import {defaultModelConfigs, ModelConfig} from '../pages/Configs';
import {Conversations} from '../pages/Chat';
import {ModelSourceItem} from '../pages/Models';
import {DownloadStatus} from '../pages/Downloads';
import {Settings} from '../pages/Settings';

export enum ModelStatus {
  Offline,
  Starting,
  Loading,
  Working,
}

class CommonStore {
  constructor() {
    makeAutoObservable(this);
  }

  // global
  modelStatus: ModelStatus = ModelStatus.Offline;

  // home
  introduction: { [lang: string]: string } = manifest.introduction;

  // chat
  conversations: Conversations = {};
  conversationsOrder: string[] = [];

  // configs
  currentModelConfigIndex: number = 0;
  modelConfigs: ModelConfig[] = [];

  // models
  modelSourceManifestList: string = 'https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner/manifest.json;';
  modelSourceList: ModelSourceItem[] = [];

  // downloads
  downloadList: DownloadStatus[] = [];

  // settings
  settings: Settings = {
    language: getUserLanguage(),
    darkMode: !isSystemLightMode(),
    autoUpdatesCheck: true
  };

  // about
  about: { [lang: string]: string } = manifest.about;

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

  setSettings = (value: Partial<Settings>, saveConfig: boolean = true) => {
    this.settings = {...this.settings, ...value};

    if (this.settings.darkMode)
      WindowSetDarkTheme();
    else
      WindowSetLightTheme();

    if (saveConfig)
      saveConfigs();
  };

  setIntroduction = (value: { [lang: string]: string }) => {
    this.introduction = value;
  };

  setAbout = (value: { [lang: string]: string }) => {
    this.about = value;
  };

  setDownloadList = (value: DownloadStatus[]) => {
    this.downloadList = value;
  };

  setConversations = (value: Conversations) => {
    this.conversations = value;
  };

  setConversationsOrder = (value: string[]) => {
    this.conversationsOrder = value;
  };
}

export default new CommonStore();