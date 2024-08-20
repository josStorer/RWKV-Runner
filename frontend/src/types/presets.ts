import { ReactElement } from 'react'
import { ConversationMessage } from './chat'

export type PresetType = 'chat' | 'completion' | 'chatInCompletion'
export type Preset = {
  name: string
  tag: string
  // if name and sourceUrl are same, it will be overridden when importing
  sourceUrl: string
  desc: string
  avatarImg: string
  userAvatarImg?: string
  type: PresetType
  // chat
  welcomeMessage: string
  messages: ConversationMessage[]
  displayPresetMessages: boolean
  // completion
  prompt: string
  stop: string
  injectStart: string
  injectEnd: string
  presystem?: boolean
  userName?: string
  assistantName?: string
}
export type PresetsNavigationItem = {
  icon: ReactElement
  element: ReactElement
}
