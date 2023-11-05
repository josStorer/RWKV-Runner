import { makeAutoObservable } from 'mobx';
import { getUserLanguage, isSystemLightMode, saveCache, saveConfigs, savePresets } from '../utils';
import { WindowSetDarkTheme, WindowSetLightTheme } from '../../wailsjs/runtime';
import manifest from '../../../manifest.json';
import { ModelConfig } from '../pages/Configs';
import { Conversation } from '../pages/Chat';
import { ModelSourceItem } from '../pages/Models';
import { DownloadStatus } from '../pages/Downloads';
import { SettingsType } from '../pages/Settings';
import { IntroductionContent } from '../pages/Home';
import { AboutContent } from '../pages/About';
import i18n from 'i18next';
import { CompletionPreset } from '../pages/Completion';
import { defaultCompositionPrompt, defaultModelConfigs, defaultModelConfigsMac } from '../pages/defaultConfigs';
import commonStore from './commonStore';
import { Preset } from '../pages/PresetsManager/PresetsButton';
import { DataProcessParameters, LoraFinetuneParameters } from '../pages/Train';
import { ChartData } from 'chart.js';
import { CompositionParams } from '../pages/Composition';

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

export type Attachment = {
  name: string;
  size: number;
  content: string;
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
  // presets manager
  editingPreset: Preset | null = null;
  presets: Preset[] = [];
  // home
  introduction: IntroductionContent = manifest.introduction;
  // chat
  currentInput: string = '';
  conversation: Conversation = {};
  conversationOrder: string[] = [];
  activePreset: Preset | null = null;
  attachmentUploading: boolean = false;
  attachments: { [uuid: string]: Attachment[] } = {};
  currentTempAttachment: Attachment | null = null;
  // completion
  completionPreset: CompletionPreset | null = null;
  completionGenerating: boolean = false;
  completionSubmittedPrompt: string = '';
  // composition
  compositionParams: CompositionParams = {
    prompt: defaultCompositionPrompt,
    maxResponseToken: 200,
    temperature: 1,
    topP: 0.8,
    autoPlay: true,
    useLocalSoundFont: false,
    midi: null,
    ns: null
  };
  compositionGenerating: boolean = false;
  compositionSubmittedPrompt: string = defaultCompositionPrompt;
  // configs
  currentModelConfigIndex: number = 0;
  modelConfigs: ModelConfig[] = [];
  modelParamsCollapsed: boolean = true;
  // models
  modelSourceManifestList: string = 'https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner@master/manifest.json;';
  modelSourceList: ModelSourceItem[] = [];
  // downloads
  downloadList: DownloadStatus[] = [];
  lastUnfinishedModelDownloads: DownloadStatus[] = [];
  // train
  wslStdout: string = '';
  chartTitle: string = '';
  chartData: ChartData<'line', (number | null)[], string> = { labels: [], datasets: [] };
  loraModels: string[] = [];
  dataProcessParams: DataProcessParameters = {
    dataPath: 'finetune/data/sample.jsonl',
    vocabPath: 'backend-python/rwkv_pip/rwkv_vocab_v20230424.txt'
  };
  loraFinetuneParams: LoraFinetuneParameters = {
    baseModel: '',
    ctxLen: 1024,
    epochSteps: 200,
    epochCount: 20,
    epochBegin: 0,
    epochSave: 2,
    microBsz: 1,
    accumGradBatches: 8,
    preFfn: false,
    headQk: false,
    lrInit: '5e-5',
    lrFinal: '5e-5',
    warmupSteps: 0,
    beta1: 0.9,
    beta2: 0.999,
    adamEps: '1e-8',
    devices: 1,
    precision: 'bf16',
    gradCp: false,
    loraR: 8,
    loraAlpha: 32,
    loraDropout: 0.01,
    loraLoad: ''
  };
  // settings
  advancedCollapsed: boolean = true;
  settings: SettingsType = {
    language: getUserLanguage(),
    darkMode: !isSystemLightMode(),
    autoUpdatesCheck: true,
    giteeUpdatesSource: getUserLanguage() === 'zh',
    cnMirror: getUserLanguage() === 'zh',
    host: '127.0.0.1',
    dpiScaling: 100,
    customModelsPath: './models',
    customPythonPath: '',
    apiUrl: '',
    apiKey: 'sk-',
    apiChatModelName: 'rwkv',
    apiCompletionModelName: 'rwkv'
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
      config = JSON.parse(JSON.stringify(commonStore.platform !== 'darwin' ? defaultModelConfigs[0] : defaultModelConfigsMac[0]));
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

  setDepComplete = (value: boolean, inSaveCache: boolean = true) => {
    this.depComplete = value;
    if (inSaveCache)
      saveCache();
  };

  setDownloadList = (value: DownloadStatus[]) => {
    this.downloadList = value;
  };

  setConversation = (value: Conversation) => {
    this.conversation = value;
  };

  setConversationOrder = (value: string[]) => {
    this.conversationOrder = value;
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

  setCurrentInput(value: string) {
    this.currentInput = value;
  }

  setAdvancedCollapsed(value: boolean) {
    this.advancedCollapsed = value;
  }

  setModelParamsCollapsed(value: boolean) {
    this.modelParamsCollapsed = value;
  }

  setLastUnfinishedModelDownloads(value: DownloadStatus[]) {
    this.lastUnfinishedModelDownloads = value;
  }

  setEditingPreset(value: Preset) {
    this.editingPreset = value;
  }

  setPresets(value: Preset[], savePreset: boolean = true) {
    this.presets = value;
    if (savePreset)
      savePresets();
  }

  setActivePreset(value: Preset) {
    this.activePreset = value;
  }

  setCompletionSubmittedPrompt(value: string) {
    this.completionSubmittedPrompt = value;
  }

  setCompositionParams(value: CompositionParams) {
    this.compositionParams = value;
  }

  setCompositionGenerating(value: boolean) {
    this.compositionGenerating = value;
  }

  setCompositionSubmittedPrompt(value: string) {
    this.compositionSubmittedPrompt = value;
  }

  setWslStdout(value: string) {
    this.wslStdout = value;
  }

  setDataProcessParams(value: DataProcessParameters, saveConfig: boolean = true) {
    this.dataProcessParams = value;
    if (saveConfig)
      saveConfigs();
  }

  setLoraFinetuneParameters(value: LoraFinetuneParameters, saveConfig: boolean = true) {
    this.loraFinetuneParams = value;
    if (saveConfig)
      saveConfigs();
  }

  setChartTitle(value: string) {
    this.chartTitle = value;
  }

  setChartData(value: ChartData<'line', (number | null)[], string>) {
    this.chartData = value;
  }

  setLoraModels(value: string[]) {
    this.loraModels = value;
  }

  setAttachmentUploading(value: boolean) {
    this.attachmentUploading = value;
  }

  setAttachments(value: { [uuid: string]: Attachment[] }) {
    this.attachments = value;
  }

  setAttachment(uuid: string, value: Attachment[] | null) {
    if (value === null)
      delete this.attachments[uuid];
    else
      this.attachments[uuid] = value;
  }

  setCurrentTempAttachment(value: Attachment | null) {
    this.currentTempAttachment = value;
  }
}

export default new CommonStore();