import { ReactElement } from 'react'

export type DataProcessParameters = {
  dataPath: string
  vocabPath: string
}
export type LoraFinetunePrecision = 'bf16' | 'fp16' | 'tf32'
export type LoraFinetuneParameters = {
  baseModel: string
  ctxLen: number
  epochSteps: number
  epochCount: number
  epochBegin: number
  epochSave: number
  microBsz: number
  accumGradBatches: number
  preFfn: boolean
  headQk: boolean
  lrInit: string
  lrFinal: string
  warmupSteps: number
  beta1: number
  beta2: number
  adamEps: string
  devices: number
  precision: LoraFinetunePrecision
  gradCp: boolean
  loraR: number
  loraAlpha: number
  loraDropout: number
  loraLoad: string
}
export type TrainNavigationItem = {
  element: ReactElement
}
