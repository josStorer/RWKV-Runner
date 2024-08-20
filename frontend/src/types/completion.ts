import { ApiParameters } from './configs'

export type CompletionParams = Omit<ApiParameters, 'apiPort'> & {
  stop: string
  injectStart: string
  injectEnd: string
}
export type CompletionPreset = {
  name: string
  prompt: string
  params: CompletionParams
}
