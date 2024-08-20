export const Languages = {
  dev: 'English', // i18n default
  zh: '简体中文',
  ja: '日本語',
}
export type Language = keyof typeof Languages
export type SettingsType = {
  language: Language
  darkMode: boolean
  autoUpdatesCheck: boolean
  giteeUpdatesSource: boolean
  cnMirror: boolean
  useHfMirror: boolean
  host: string
  dpiScaling: number
  customModelsPath: string
  customPythonPath: string
  apiUrl: string
  apiKey: string
  apiChatModelName: string
  apiCompletionModelName: string
  coreApiUrl: string
}
