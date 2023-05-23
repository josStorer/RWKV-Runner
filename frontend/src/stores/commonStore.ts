import { makeAutoObservable } from 'mobx';
import { getUserLanguage, isSystemLightMode, saveConfigs } from '../utils';
import { WindowSetDarkTheme, WindowSetLightTheme } from '../../wailsjs/runtime';
import manifest from '../../../manifest.json';
import { defaultModelConfigs, ModelConfig } from '../pages/Configs';
import { Conversations } from '../pages/Chat';
import { ModelSourceItem } from '../pages/Models';
import { DownloadStatus } from '../pages/Downloads';
import { SettingsType } from '../pages/Settings';
import { IntroductionContent } from '../pages/Home';
import { AboutContent } from '../pages/About';
import i18n from 'i18next';

export enum ModelStatus {
  Offline,
  Starting,
  Loading,
  Working,
}

export type Status = {
  modelStatus: ModelStatus;
  pid: number;
  device_name: string;
}

class CommonStore {
  // global
  status: Status = {
    modelStatus: ModelStatus.Offline,
    pid: 0,
    device_name: 'CPU'
  };
  depComplete: boolean = false;
  // home
  introduction: IntroductionContent = manifest.introduction;
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
  settings: SettingsType = {
    language: getUserLanguage(),
    darkMode: !isSystemLightMode(),
    autoUpdatesCheck: true,
    giteeUpdatesSource: getUserLanguage() === 'zh',
    cnMirror: getUserLanguage() === 'zh'
  };
  // about
  about: AboutContent = manifest.about;

  constructor() {
    makeAutoObservable(this);
  }

  getCurrentModelConfig = () => {
    return this.modelConfigs[this.currentModelConfigIndex];
  };

  setStatus = (status: Partial<Status>) => {
    this.status = { ...this.status, ...status };
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

  setSettings = (value: Partial<SettingsType>, saveConfig: boolean = true) => {
    this.settings = { ...this.settings, ...value };

    if (this.settings.darkMode)
      WindowSetDarkTheme();
    else
      WindowSetLightTheme();

    if (this.settings.language)
      i18n.changeLanguage(this.settings.language);

    if (saveConfig)
      saveConfigs();
  };

  setIntroduction = (value: IntroductionContent) => {
    this.introduction = value;
  };

  setAbout = (value: AboutContent) => {
    this.about = value;
  };

  setDepComplete = (value: boolean) => {
    this.depComplete = value;
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