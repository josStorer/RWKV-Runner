import { makeAutoObservable } from 'mobx';
import { getUserLanguage, isSystemLightMode, saveConfigs } from '../utils';
import { WindowSetDarkTheme, WindowSetLightTheme } from '../../wailsjs/runtime';
import manifest from '../../../manifest.json';
import { ModelConfig } from '../pages/Configs';
import { Conversations } from '../pages/Chat';
import { ModelSourceItem } from '../pages/Models';
import { DownloadStatus } from '../pages/Downloads';
import { SettingsType } from '../pages/Settings';
import { IntroductionContent } from '../pages/Home';
import { AboutContent } from '../pages/About';
import i18n from 'i18next';
import { CompletionPreset } from '../pages/Completion';
import { defaultModelConfigs, defaultModelConfigsMac } from '../pages/defaultModelConfigs';
import commonStore from './commonStore';

export enum ModelStatus {
  Offline,
  Starting,
  Loading,
  Working,
}

export type Status = {
  status: ModelStatus;
  pid: number;
  device_name: string;
}

export type Platform = 'windows' | 'darwin' | 'linux';

class CommonStore {
  // global
  status: Status = {
    status: ModelStatus.Offline,
    pid: 0,
    device_name: 'CPU'
  };
  depComplete: boolean = false;
  platform: Platform = 'windows';
  // home
  introduction: IntroductionContent = manifest.introduction;
  // chat
  conversations: Conversations = {};
  conversationsOrder: string[] = [];
  // completion
  completionPreset: CompletionPreset | null = null;
  completionGenerating: boolean = false;
  // configs
  currentModelConfigIndex: number = 0;
  modelConfigs: ModelConfig[] = [];
  // models
  modelSourceManifestList: string = 'https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner@master/manifest.json;';
  modelSourceList: ModelSourceItem[] = [];
  // downloads
  downloadList: DownloadStatus[] = [];
  // settings
  settings: SettingsType = {
    language: getUserLanguage(),
    darkMode: !isSystemLightMode(),
    autoUpdatesCheck: true,
    giteeUpdatesSource: getUserLanguage() === 'zh',
    cnMirror: getUserLanguage() === 'zh',
    host: '127.0.0.1',
    customModelsPath: './models',
    customPythonPath: ''
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
    this.modelConfigs = JSON.parse(JSON.stringify(configs)); // deep copy
    if (saveConfig)
      saveConfigs();
  };

  createModelConfig = (config: ModelConfig = defaultModelConfigs[0], saveConfig: boolean = true) => {
    if (config.name === defaultModelConfigs[0].name) {
      // deep copy
      config = JSON.parse(JSON.stringify(commonStore.platform != 'darwin' ? defaultModelConfigs[0] : defaultModelConfigsMac[0]));
      config.name = new Date().toLocaleString();
    }
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

  setCompletionPreset(value: CompletionPreset) {
    this.completionPreset = value;
  }

  setCompletionGenerating(value: boolean) {
    this.completionGenerating = value;
  }

  setPlatform(value: Platform) {
    this.platform = value;
  }
}

export default new CommonStore();