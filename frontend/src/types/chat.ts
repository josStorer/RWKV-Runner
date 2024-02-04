import { ApiParameters } from './configs';

export const userName = 'M E';
export const botName = 'A I';
export const welcomeUuid = 'welcome';

export enum MessageType {
  Normal,
  Error
}

export type Side = 'left' | 'right'
export type Color = 'neutral' | 'brand' | 'colorful'
export type MessageItem = {
  sender: string,
  type: MessageType,
  color: Color,
  avatarImg?: string,
  time: string,
  content: string,
  side: Side,
  done: boolean
}
export type Conversation = {
  [uuid: string]: MessageItem
}
export type Role = 'assistant' | 'user' | 'system';
export type ConversationMessage = {
  role: Role;
  content: string;
}
export type Attachment = {
  name: string;
  size: number;
  content: string;
}
export type ChatParams = Omit<ApiParameters, 'apiPort'> & {
  markdown: boolean
}