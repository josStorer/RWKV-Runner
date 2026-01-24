import { ChartData } from 'chart.js'
import i18n from 'i18next'
import { makeAutoObservable } from 'mobx'
import manifest from '../../../manifest.json'
import { GetTorchVersion } from '../../wailsjs/go/backend_golang/App'
import { WindowSetDarkTheme, WindowSetLightTheme } from '../../wailsjs/runtime'
import {
  defaultCompositionPrompt,
  defaultModelConfigs,
  defaultModelConfigsMac,
  defaultPenaltyDecay,
} from '../pages/defaultConfigs'
import { AboutContent } from '../types/about'
import { Attachment, ChatParams, Conversation } from '../types/chat'
import { CompletionPreset } from '../types/completion'
import {
  CompositionParams,
  InstrumentType,
  MidiMessage,
  MidiPort,
  Track,
  tracksMinimalTotalTime,
} from '../types/composition'
import { ModelConfig } from '../types/configs'
import { DownloadStatus } from '../types/downloads'
import { IntroductionContent } from '../types/home'
import { ModelSourceItem } from '../types/models'
import { Preset } from '../types/presets'
import { SettingsType } from '../types/settings'
import { DataProcessParameters, LoraFinetuneParameters } from '../types/train'
import {
  getUserLanguage,
  isSystemLightMode,
  saveCache,
  saveConfigs,
  saveDurableData,
  savePresets,
} from '../utils'
import { FilterFunctionProperties } from '../utils/filter-function-properties'

export enum ModelStatus {
  Offline,
  Starting,
  Loading,
  Working,
}

export type Status = {
  status: ModelStatus
  pid: number
  device_name: string
}

export type GpuType = 'Nvidia' | 'Amd' | 'Intel'

export type MonitorData = {
  gpuType: GpuType
  gpuName: String
  usedMemory: number
  totalMemory: number
  gpuUsage: number
  usedVram: number
  totalVram: number
}

export type Platform = 'windows' | 'darwin' | 'linux' | 'web'

export type CommonStoreType = FilterFunctionProperties<
  InstanceType<typeof CommonStore>
>

export type CommonStorePropertyKey = keyof CommonStoreType

class CommonStore {
  // global
  status: Status = {
    status: ModelStatus.Offline,
    pid: 0,
    device_name: 'CPU',
  }
  monitorData: MonitorData | null = null
  depComplete: boolean = false
  platform: Platform = 'windows'
  cudaComputeCapability: string = ''
  driverCudaVersion: string = ''
  torchVersion: string = ''
  proxyPort: number = 0
  lastModelName: string = ''
  stateModels: string[] = []
  // presets manager
  editingPreset: Preset | null = null
  presets: Preset[] = []
  // home
  introduction: IntroductionContent = manifest.introduction
  // chat
  currentInput: string = ''
  conversation: Conversation = {}
  conversationOrder: string[] = []
  quickThink: boolean = false
  deepThink: boolean = false
  activePreset: Preset | null = null
  activePresetIndex: number = -1
  attachmentUploading: boolean = false
  attachments: {
    [uuid: string]: Attachment[]
  } = {}
  currentTempAttachment: Attachment | null = null
  chatParams: ChatParams = {
    maxResponseToken: 2000,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0,
    frequencyPenalty: 1,
    penaltyDecay: defaultPenaltyDecay,
    historyN: 0,
    markdown: true,
    functionCall: false,
    toolDefinition:
      '{\n' +
      '    "name": "get_current_weather",\n' +
      '    "description": "Get the current weather in a given location",\n' +
      '    "parameters": {\n' +
      '        "type": "object",\n' +
      '        "properties": {\n' +
      '            "location": {\n' +
      '                "type": "string",\n' +
      '                "description": "The city and state, e.g. San Francisco, CA"\n' +
      '            },\n' +
      '            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}\n' +
      '        },\n' +
      '        "required": ["location"]\n' +
      '    }\n' +
      '}',
    toolReturn: '{"location": "Paris", "temperature": "22"}',
  }
  sidePanelCollapsed: boolean | 'auto' = 'auto'
  screenshotting: boolean = false
  // completion
  completionPreset: CompletionPreset | null = null
  completionGenerating: boolean = false
  completionSubmittedPrompt: string = ''
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
    playOnlyGeneratedContent: true,
  }
  compositionGenerating: boolean = false
  compositionSubmittedPrompt: string = defaultCompositionPrompt
  // composition midi device
  midiPorts: MidiPort[] = []
  activeMidiDeviceIndex: number = -1
  instrumentType: InstrumentType = InstrumentType.Piano
  // composition tracks
  tracks: Track[] = []
  trackScale: number = 1
  trackTotalTime: number = tracksMinimalTotalTime
  trackCurrentTime: number = 0
  trackPlayStartTime: number = 0
  playingTrackId: string = ''
  recordingTrackId: string = ''
  recordingContent: string = '' // used to improve performance of midiMessageHandler, and I'm too lazy to maintain an ID dictionary for this (although that would be better for realtime effects)
  recordingRawContent: MidiMessage[] = []
  // configs
  autoConfigPort?: number
  currentModelConfigIndex: number = 0
  modelConfigs: ModelConfig[] = []
  apiParamsCollapsed: boolean = true
  modelParamsCollapsed: boolean = true
  // models
  activeModelListTags: string[] = []
  modelSourceManifestList: string =
    'https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner@master/manifest.json;'
  modelSourceList: ModelSourceItem[] = []
  // downloads
  downloadList: DownloadStatus[] = []
  lastUnfinishedModelDownloads: DownloadStatus[] = []
  // train
  wslStdout: string = ''
  chartTitle: string = ''
  chartData: ChartData<'line', (number | null)[], string> = {
    labels: [],
    datasets: [],
  }
  loraModels: string[] = []
  dataProcessParams: DataProcessParameters = {
    dataPath: 'finetune/data/sample.jsonl',
    vocabPath: 'backend-python/rwkv_pip/rwkv_vocab_v20230424.txt',
  }
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
    gradCp: true,
    loraR: 64,
    loraAlpha: 192,
    loraDropout: 0.01,
    loraLoad: '',
  }
  // settings
  advancedCollapsed: boolean = true
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
    coreApiUrl: '',
    rememberAllDurableData: true,
  }
  // about
  about: AboutContent = manifest.about

  constructor() {
    makeAutoObservable(this)
  }

  setData(data: Partial<CommonStoreType>, saveConfig: boolean = true) {
    Object.assign(this, data)

    if (saveConfig) saveDurableData()
  }

  get customKernelSupported() {
    return (
      this.platform !== 'windows' ||
      !this.torchVersion ||
      this.torchVersion === '1.13.1+cu117' ||
      this.torchVersion === '2.7.1+cu128'
    )
  }

  setAutoConfigPort(port?: number) {
    this.autoConfigPort = port
  }

  getCurrentModelConfig = () => {
    return this.modelConfigs[this.currentModelConfigIndex]
  }

  setStatus = (status: Partial<Status>) => {
    this.status = { ...this.status, ...status }
  }

  setCurrentConfigIndex = (index: number, saveConfig: boolean = true) => {
    this.currentModelConfigIndex = index
    if (saveConfig) saveConfigs()
  }

  setModelConfig = (
    index: number,
    config: ModelConfig,
    saveConfig: boolean = true
  ) => {
    this.modelConfigs[index] = config
    if (saveConfig) saveConfigs()
  }

  setModelConfigs = (configs: ModelConfig[], saveConfig: boolean = true) => {
    this.modelConfigs = JSON.parse(JSON.stringify(configs)) // deep copy
    if (saveConfig) saveConfigs()
  }

  createModelConfig = (
    config: ModelConfig = defaultModelConfigs[0],
    saveConfig: boolean = true
  ) => {
    if (config.name === defaultModelConfigs[0].name) {
      // deep copy
      config = JSON.parse(
        JSON.stringify(
          this.platform !== 'darwin'
            ? defaultModelConfigs[0]
            : defaultModelConfigsMac[0]
        )
      )
      config.name = new Date().toLocaleString()
    }
    this.modelConfigs.push(config)
    if (saveConfig) saveConfigs()
  }

  deleteModelConfig = (index: number, saveConfig: boolean = true) => {
    this.modelConfigs.splice(index, 1)
    if (index < this.currentModelConfigIndex) {
      this.setCurrentConfigIndex(this.currentModelConfigIndex - 1)
    }
    if (this.modelConfigs.length === 0) {
      this.createModelConfig()
    }
    if (this.currentModelConfigIndex >= this.modelConfigs.length) {
      this.setCurrentConfigIndex(this.modelConfigs.length - 1)
    }
    if (saveConfig) saveConfigs()
  }

  setModelSourceManifestList = (value: string, saveConfig: boolean = true) => {
    this.modelSourceManifestList = value
    if (saveConfig) saveConfigs()
  }

  setModelSourceList = (value: ModelSourceItem[]) => {
    this.modelSourceList = value
  }

  setSettings = (value: Partial<SettingsType>, saveConfig: boolean = true) => {
    this.settings = { ...this.settings, ...value }

    if (this.settings.darkMode) {
      WindowSetDarkTheme()
      document.documentElement.setAttribute('style', 'color-scheme: dark;')
    } else {
      WindowSetLightTheme()
      document.documentElement.setAttribute('style', 'color-scheme: light;')
    }

    if (this.settings.language) {
      i18n.changeLanguage(this.settings.language)
      document.documentElement.setAttribute(
        'lang',
        this.settings.language === 'dev' ? 'en' : this.settings.language
      )
    }

    if (saveConfig) saveConfigs()
  }

  setIntroduction = (value: IntroductionContent) => {
    this.introduction = value
  }

  setAbout = (value: AboutContent) => {
    this.about = value
  }

  setLastModelName(value: string) {
    this.lastModelName = value
  }

  setDepComplete = (value: boolean, inSaveCache: boolean = true) => {
    this.depComplete = value
    if (inSaveCache) saveCache()
  }

  setDownloadList = (value: DownloadStatus[]) => {
    this.downloadList = value
  }

  setConversation = (value: Conversation) => {
    this.conversation = value
    saveDurableData()
  }

  setConversationOrder = (value: string[]) => {
    this.conversationOrder = value
    saveDurableData()
  }

  setCompletionPreset(value: CompletionPreset) {
    this.completionPreset = value
    saveDurableData()
  }

  setCompletionGenerating(value: boolean) {
    this.completionGenerating = value
  }

  setMonitorData(value: MonitorData) {
    this.monitorData = value
  }

  setPlatform(value: Platform) {
    this.platform = value
  }

  setCudaComputeCapability(value: string) {
    this.cudaComputeCapability = value
  }

  setDriverCudaVersion(value: string) {
    this.driverCudaVersion = value
  }

  refreshTorchVersion() {
    GetTorchVersion(this.settings.customPythonPath).then((v) => {
      this.torchVersion = v
    })
  }

  setProxyPort(value: number) {
    this.proxyPort = value
  }

  setCurrentInput(value: string) {
    this.currentInput = value
    saveDurableData()
  }

  setQuickThink(value: boolean) {
    this.quickThink = value
    saveDurableData()
  }

  setDeepThink(value: boolean) {
    this.deepThink = value
    saveDurableData()
  }

  setAdvancedCollapsed(value: boolean) {
    this.advancedCollapsed = value
    saveDurableData()
  }

  setApiParamsCollapsed(value: boolean) {
    this.apiParamsCollapsed = value
    saveDurableData()
  }

  setModelParamsCollapsed(value: boolean) {
    this.modelParamsCollapsed = value
    saveDurableData()
  }

  setLastUnfinishedModelDownloads(value: DownloadStatus[]) {
    this.lastUnfinishedModelDownloads = value
  }

  setEditingPreset(value: Preset | null) {
    this.editingPreset = value
  }

  setPresets(value: Preset[], savePreset: boolean = true) {
    this.presets = value
    if (savePreset) savePresets()
  }

  setActivePreset(value: Preset | null) {
    this.activePreset = value
    saveDurableData()
  }

  setActivePresetIndex(value: number) {
    this.activePresetIndex = value
    saveDurableData()
  }

  setCompletionSubmittedPrompt(value: string) {
    this.completionSubmittedPrompt = value
    saveDurableData()
  }

  setCompositionParams(value: CompositionParams, saveDurable: boolean = true) {
    this.compositionParams = value
    if (saveDurable) saveDurableData()
  }

  setCompositionGenerating(value: boolean) {
    this.compositionGenerating = value
  }

  setCompositionSubmittedPrompt(value: string) {
    this.compositionSubmittedPrompt = value
    saveDurableData()
  }

  setWslStdout(value: string) {
    this.wslStdout = value
  }

  setDataProcessParams(
    value: DataProcessParameters,
    saveConfig: boolean = true
  ) {
    this.dataProcessParams = value
    if (saveConfig) saveConfigs()
  }

  setLoraFinetuneParameters(
    value: LoraFinetuneParameters,
    saveConfig: boolean = true
  ) {
    this.loraFinetuneParams = value
    if (saveConfig) saveConfigs()
  }

  setChartTitle(value: string) {
    this.chartTitle = value
  }

  setChartData(value: ChartData<'line', (number | null)[], string>) {
    this.chartData = value
  }

  setLoraModels(value: string[]) {
    this.loraModels = value
  }

  setStateModels(value: string[]) {
    this.stateModels = value
  }

  setAttachmentUploading(value: boolean) {
    this.attachmentUploading = value
  }

  setAttachments(value: { [uuid: string]: Attachment[] }) {
    this.attachments = value
    saveDurableData()
  }

  setAttachment(uuid: string, value: Attachment[] | null) {
    if (value === null) delete this.attachments[uuid]
    else this.attachments[uuid] = value
    saveDurableData()
  }

  setCurrentTempAttachment(value: Attachment | null) {
    this.currentTempAttachment = value
    saveDurableData()
  }

  setChatParams(value: Partial<ChatParams>) {
    this.chatParams = { ...this.chatParams, ...value }
    saveDurableData()
  }

  setSidePanelCollapsed(value: boolean | 'auto') {
    this.sidePanelCollapsed = value
    saveDurableData()
  }

  setScreenshotting(value: boolean) {
    this.screenshotting = value
  }

  setTracks(value: Track[]) {
    this.tracks = value
    saveDurableData()
  }

  setTrackScale(value: number) {
    this.trackScale = value
    saveDurableData()
  }

  setTrackTotalTime(value: number) {
    this.trackTotalTime = value
    saveDurableData()
  }

  setTrackCurrentTime(value: number) {
    this.trackCurrentTime = value
    saveDurableData()
  }

  setTrackPlayStartTime(value: number) {
    this.trackPlayStartTime = value
    saveDurableData()
  }

  setMidiPorts(value: MidiPort[]) {
    this.midiPorts = value
  }

  setInstrumentType(value: InstrumentType) {
    this.instrumentType = value
    saveDurableData()
  }

  setRecordingTrackId(value: string) {
    this.recordingTrackId = value
  }

  setActiveMidiDeviceIndex(value: number) {
    this.activeMidiDeviceIndex = value
  }

  setRecordingContent(value: string) {
    this.recordingContent = value
  }

  setRecordingRawContent(value: MidiMessage[]) {
    this.recordingRawContent = value
  }

  setPlayingTrackId(value: string) {
    this.playingTrackId = value
  }

  setActiveModelListTags(value: string[]) {
    this.activeModelListTags = value
    saveDurableData()
  }
}

export default new CommonStore()
