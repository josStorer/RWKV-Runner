export type ApiParameters = {
  apiPort: number
  maxResponseToken: number
  temperature: number
  topP: number
  presencePenalty: number
  frequencyPenalty: number
  penaltyDecay?: number
  globalPenalty?: boolean
  stateModel?: string
}
export type Device =
  | 'CPU'
  | 'CPU (rwkv.cpp)'
  | 'CUDA'
  | 'CUDA-Beta'
  | 'WebGPU'
  | 'WebGPU (Python)'
  | 'MPS'
  | 'Custom'
export type Precision = 'fp16' | 'int8' | 'fp32' | 'nf4' | 'Q5_1'
export type ModelParameters = {
  // different models can not have the same name
  modelName: string
  device: Device
  precision: Precision
  storedLayers: number
  maxStoredLayers: number
  quantizedLayers?: number
  tokenChunkSize?: number
  useCustomCuda?: boolean
  customStrategy?: string
  useCustomTokenizer?: boolean
  customTokenizer?: string
}
export type ModelConfig = {
  // different configs can have the same name
  name: string
  apiParameters: ApiParameters
  modelParameters: ModelParameters
  enableWebUI?: boolean
}
