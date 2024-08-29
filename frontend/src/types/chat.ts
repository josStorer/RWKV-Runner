import { ApiParameters } from './configs'

export const userName = 'M E'
export const botName = 'A I'
export const systemName = 'System'
export const welcomeUuid = 'welcome'

export enum MessageType {
  Normal,
  Error,
}

export type Side = 'left' | 'right' | 'center'
export type Color = 'neutral' | 'brand' | 'colorful'
export type MessageItem = {
  sender: string
  toolName?: string
  type: MessageType
  color: Color
  avatarImg?: string
  time: string
  content: string
  side: Side
  done: boolean
}
export type Conversation = {
  [uuid: string]: MessageItem
}
export type Role = 'assistant' | 'user' | 'system' | 'tool'
export type ConversationMessage = {
  role: Role
  content: string
  tool_call_id?: string
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: {
      name: string
      arguments: string
    }
  }>
}
export type Attachment = {
  name: string
  size: number
  content: string
}
export type ChatParams = Omit<ApiParameters, 'apiPort'> & {
  historyN: number
  markdown: boolean
  functionCall: boolean
  toolDefinition: string
  toolReturn: string
}
