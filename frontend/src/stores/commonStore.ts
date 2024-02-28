import { makeAutoObservable } from 'mobx';
import { getUserLanguage, isSystemLightMode, saveCache, saveConfigs, savePresets } from '../utils';
import { WindowSetDarkTheme, WindowSetLightTheme } from '../../wailsjs/runtime';
import manifest from '../../../manifest.json';
import i18n from 'i18next';
import {
  defaultCompositionPrompt,
  defaultModelConfigs,
  defaultModelConfigsMac,
  defaultPenaltyDecay
} from '../pages/defaultConfigs';
import { ChartData } from 'chart.js';
import { Preset } from '../types/presets';
import { AboutContent } from '../types/about';
import { Attachment, ChatParams, Conversation } from '../types/chat';
import { CompletionPreset } from '../types/completion';
import {
  CompositionParams,
  InstrumentType,
  MidiMessage,
  MidiPort,
  Track,
  tracksMinimalTotalTime
} from '../types/composition';
import { ModelConfig } from '../types/configs';
import { DownloadStatus } from '../types/downloads';
import { IntroductionContent } from '../types/home';
import { ModelSourceItem } from '../types/models';
import { SettingsType } from '../types/settings';
import { DataProcessParameters, LoraFinetuneParameters } from '../types/train';

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

export type MonitorData = {
  usedMemory: number;
  totalMemory: number;
  gpuUsage: number;
  gpuPower: number;
  usedVram: number;
  totalVram: number;
}

export type Platform = 'windows' | 'darwin' | 'linux' | 'web';

class CommonStore {
  // global
  status: Status = {
    status: ModelStatus.Offline,
    pid: 0,
    device_name: 'CPU'
  };
  monitorData: MonitorData | null = null;
  depComplete: boolean = false;
  platform: Platform = 'windows';
  lastModelName: string = '';
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
  attachments: {
    [uuid: string]: Attachment[]
  } = {};
  currentTempAttachment: Attachment | null = null;
  chatParams: ChatParams = {
    maxResponseToken: 1000,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0,
    frequencyPenalty: 1,
    penaltyDecay: defaultPenaltyDecay,
    historyN: 0,
    markdown: true
  };
  sidePanelCollapsed: boolean | 'auto' = 'auto';
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
    externalPlay: false,
    midi: null,
    ns: null,
    generationStartTime: 0,
    playOnlyGeneratedContent: true
  };
  compositionGenerating: boolean = false;
  compositionSubmittedPrompt: string = defaultCompositionPrompt;
  // composition midi device
  midiPorts: MidiPort[] = [];
  activeMidiDeviceIndex: number = -1;
  instrumentType: InstrumentType = InstrumentType.Piano;
  // composition tracks
  tracks: Track[] = [];
  trackScale: number = 1;
  trackTotalTime: number = tracksMinimalTotalTime;
  trackCurrentTime: number = 0;
  trackPlayStartTime: number = 0;
  playingTrackId: string = '';
  recordingTrackId: string = '';
  recordingContent: string = ''; // used to improve performance of midiMessageHandler, and I'm too lazy to maintain an ID dictionary for this (although that would be better for realtime effects)
  recordingRawContent: MidiMessage[] = [];
  // configs
  currentModelConfigIndex: number = 0;
  modelConfigs: ModelConfig[] = [];
  modelParamsCollapsed: boolean = true;
  // models
  activeModelListTags: string[] = [];
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
    epochSave: 1,
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
    useHfMirror: getUserLanguage() === 'zh',
    host: '127.0.0.1',
    dpiScaling: 100,
    customModelsPath: './models',
    customPythonPath: '',
    apiUrl: '',
    apiKey: '',
    apiChatModelName: 'rwkv',
    apiCompletionModelName: 'rwkv',
    coreApiUrl: ''
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
      config = JSON.parse(JSON.stringify(this.platform !== 'darwin' ? defaultModelConfigs[0] : defaultModelConfigsMac[0]));
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

  setLastModelName(value: string) {
    this.lastModelName = value;
  }

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

  setMonitorData(value: MonitorData) {
    this.monitorData = value;
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

  setActivePreset(value: Preset | null) {
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

  setAttachments(value: {
    [uuid: string]: Attachment[]
  }) {
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

  setChatParams(value: Partial<ChatParams>) {
    this.chatParams = { ...this.chatParams, ...value };
  }

  setSidePanelCollapsed(value: boolean | 'auto') {
    this.sidePanelCollapsed = value;
  }

  setTracks(value: Track[]) {
    this.tracks = value;
  }

  setTrackScale(value: number) {
    this.trackScale = value;
  }

  setTrackTotalTime(value: number) {
    this.trackTotalTime = value;
  }

  setTrackCurrentTime(value: number) {
    this.trackCurrentTime = value;
  }

  setTrackPlayStartTime(value: number) {
    this.trackPlayStartTime = value;
  }

  setMidiPorts(value: MidiPort[]) {
    this.midiPorts = value;
  }

  setInstrumentType(value: InstrumentType) {
    this.instrumentType = value;
  }

  setRecordingTrackId(value: string) {
    this.recordingTrackId = value;
  }

  setActiveMidiDeviceIndex(value: number) {
    this.activeMidiDeviceIndex = value;
  }

  setRecordingContent(value: string) {
    this.recordingContent = value;
  }

  setRecordingRawContent(value: MidiMessage[]) {
    this.recordingRawContent = value;
  }

  setPlayingTrackId(value: string) {
    this.playingTrackId = value;
  }

  setActiveModelListTags(value: string[]) {
    this.activeModelListTags = value;
  }
}

export default new CommonStore();