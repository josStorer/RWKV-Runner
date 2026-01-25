import { ApiParameters } from './configs'

export type StopItem = {
  type: 'text' | 'token'
  value: string
}

export type CompletionParams = Omit<ApiParameters, 'apiPort'> & {
  stopItems: StopItem[]
  stop?: string
  injectStart: string
  injectEnd: string
}
export type CompletionPreset = {
  name: string
  prompt: string
  params: CompletionParams
}
