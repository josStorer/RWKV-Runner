import { GpuType, Platform } from '../stores/commonStore'
import { Device, Precision } from '../types/configs'
import { ModelSourceItem } from '../types/models'

type SelectModelOptions = {
  hardwareInfo?: {
    platform?: Platform
    gpuType?: GpuType
    usedMemory?: number
    totalMemory?: number
    usedVram?: number
    totalVram?: number
  }
  availableModels: ModelSourceItem[]
}

type Resolution = {
  device?: Device
  model?: ModelSourceItem
  strategy?: any
  comments?: string
}

export function selectModel(options: SelectModelOptions): Resolution[] {
  const { hardwareInfo, availableModels } = options

  const platform = hardwareInfo?.platform
  const gpuType = hardwareInfo?.gpuType
  const usedMemory = hardwareInfo?.usedMemory
  const totalMemory = hardwareInfo?.totalMemory
  const usedVram = hardwareInfo?.usedVram
  const totalVram = hardwareInfo?.totalVram

  return []
}

function filterModels(options: SelectModelOptions): ModelSourceItem[] {
  const { hardwareInfo, availableModels } = options

  const platform = hardwareInfo?.platform
  const gpuType = hardwareInfo?.gpuType
  const usedMemory = hardwareInfo?.usedMemory
  const totalMemory = hardwareInfo?.totalMemory
  const usedVram = hardwareInfo?.usedVram
  const totalVram = hardwareInfo?.totalVram

  var availableVram: number | undefined = undefined
  var availableMemory: number | undefined = undefined

  var sizeLimitation: number | undefined = undefined

  if (totalVram !== undefined && usedVram !== undefined) {
    availableVram = totalVram - usedVram
    sizeLimitation = availableVram
  }

  if (
    sizeLimitation !== undefined &&
    totalMemory !== undefined &&
    usedMemory !== undefined
  ) {
    availableMemory = totalMemory - usedMemory
    sizeLimitation = availableMemory
  }

  return []
}
