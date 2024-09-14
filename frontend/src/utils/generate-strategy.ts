import commonStore from '../stores/commonStore'
import { ModelConfig, ModelParameters, Precision } from '../types/configs'

export type CompositionMode = 'MIDI' | 'ABC'

export type CPUOrGPU = 'GPU' | 'CPU'

export type Func = 'Chat' | 'Completion' | 'Composition' | 'Function Call'

enum RecommendLevel {
  Recommended = 2,
  Easy = 1,
  Unknown = 0,
  Occupy = -1,
  NotAvailable = -2,
}

/** Hard-coded model name map */
const availableModelPairs: Record<Func, string[]> = {
  Chat: [
    'RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
    'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
    'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
    'RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth',
  ],
  Completion: [
    'RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
    'RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
    'RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
    'RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth',
  ],
  Composition: [
    'RWKV-5-MIDI-120M-v1-20230728-ctx4096.pth',
    'RWKV-5-MIDI-560M-v1-20230902-ctx4096.pth',
    'RWKV-5-ABC-82M-v1-20230901-ctx1024.pth',
  ],
  'Function Call': ['Mobius-r6-chat-CHNtuned-12b-16k-v5.5.pth'],
}

const sourceConfig: ModelConfig = {
  name: '',
  enableWebUI: false,
  apiParameters: {
    apiPort: 8080,
    maxResponseToken: 4100,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0,
    frequencyPenalty: 1,
  },
  modelParameters: {
    modelName: '',
    device: 'WebGPU',
    precision: 'nf4',
    storedLayers: 41,
    maxStoredLayers: 8,
  },
}

export const generateStrategy = (
  func?: Func | null,
  compositionMode?: CompositionMode | null,
  CPUOrGPU?: CPUOrGPU | null
) => {
  if (!func || !CPUOrGPU) return []
  if (func === 'Composition' && !compositionMode) return []

  const userSelectGPU = CPUOrGPU === 'GPU'

  const {
    gpuType,
    usedMemory: usedMemoryGB,
    totalMemory: totalMemoryGB,
    usedVram: usedVramMB,
    totalVram: totalVramMB,
  } = commonStore.monitorData || {}

  const availableMemoryGB =
    totalMemoryGB !== undefined && usedMemoryGB !== undefined
      ? totalMemoryGB - usedMemoryGB
      : undefined

  const totalVramGB = totalVramMB !== undefined ? totalVramMB / 1024 : undefined
  const usedVramGB = usedVramMB !== undefined ? usedVramMB / 1024 : undefined

  const availableVramGB =
    totalVramGB !== undefined && usedVramGB !== undefined
      ? totalVramGB - usedVramGB
      : undefined

  let resolutions: Resolution[] = []

  let availableModels = func ? availableModelPairs[func] : []
  if (compositionMode && func === 'Composition') {
    availableModels = availableModels.filter((modelName) =>
      modelName.includes(compositionMode)
    )
  }

  const resolutionSource: Resolution[] = availableModels.map((modelName) => {
    const model = commonStore.modelSourceList.find(
      (item) => item.name === modelName
    )

    const fileSizeInGB = model!.size / 1024 / 1024 / 1024
    const ramRequirement_nf4 = fileSizeInGB / 3 + 1.5
    const ramRequirement_int8 = fileSizeInGB / 2 + 1.5
    const ramRequirement_Q5_1 = fileSizeInGB / 2 + 1.5
    const ramRequirement_fp16 = fileSizeInGB / 1 + 1.5
    const ramRequirement_fp32 = fileSizeInGB / 0.5 + 1.5

    return {
      modelName: modelName,
      fileSize: model!.size,
      available: true,
      recommendLevel: RecommendLevel.Unknown,
      usingGPU: false,
      calculateByPrecision: 'nf4',
      requirements: {
        fp16: ramRequirement_fp16,
        int8: ramRequirement_int8,
        fp32: ramRequirement_fp32,
        nf4: ramRequirement_nf4,
        Q5_1: ramRequirement_Q5_1,
      },
    }
  })

  const cudaAvailable = gpuType == 'Nvidia'

  const commentRAMFailed = '获取内存状态错误'
  const commentVRAMFailed = '获取显存状态错误'
  const commentEasy = '您的硬件可以轻松运行此配置'
  const commentRecommanded = '⭐我们推荐你运行此配置'
  const commentRAMOccupy =
    '⚠你的硬件理论上足够运行此配置，但由于资源占用，目前不建议使用，考虑释放内存占用后，再运行此配置'
  const commentVRAMOccupy =
    '⚠你的硬件理论上足够运行此配置，但由于资源占用，目前不建议使用，考虑释放显存占用后，再运行此配置'
  const commentNotAvailable = '❌您的硬件无法运行此配置'

  for (let i = 0; i < resolutionSource.length; i++) {
    const resolution = resolutionSource[i]

    if (!userSelectGPU) {
      const limitation =
        func === 'Composition'
          ? resolution.requirements.fp32
          : resolution.requirements.Q5_1

      let calculateByPrecision: Precision =
        func === 'Composition' ? 'fp32' : 'Q5_1'
      let comments = commentRAMFailed
      let available = true
      let recommendLevel = -1

      let modelParameters: ModelParameters = {
        modelName: resolution.modelName,
        device: func === 'Composition' ? 'CPU' : 'CPU (rwkv.cpp)',
        precision: calculateByPrecision,
        storedLayers: 41,
        maxStoredLayers: 8,
      }

      if (availableMemoryGB && availableMemoryGB > limitation) {
        available = true
        comments = commentEasy
        recommendLevel = 1
      } else if (totalMemoryGB && totalMemoryGB >= limitation) {
        available = true
        comments = commentRAMOccupy
        recommendLevel = -1
      } else if (totalMemoryGB && totalMemoryGB < limitation) {
        available = false
        comments = commentNotAvailable
        recommendLevel = -2
      } else {
        // ERROR?
      }

      const newResolution: Resolution = {
        ...resolution,
        comments,
        available,
        recommendLevel,
        calculateByPrecision,
        modelConfig: { ...sourceConfig, modelParameters },
      }
      resolutions.push(newResolution)
      continue
    }

    if (cudaAvailable && userSelectGPU) {
      const limitation = resolution.requirements.fp16

      let comments = commentVRAMFailed
      let available = true
      let recommendLevel = -1
      let calculateByPrecision: Precision = 'fp16'

      let modelParameters: ModelParameters = {
        modelName: resolution.modelName,
        device: 'CUDA',
        precision: calculateByPrecision,
        storedLayers: 41,
        maxStoredLayers: 8,
        useCustomCuda: true,
      }

      if (availableVramGB && availableVramGB > limitation) {
        available = true
        comments = commentEasy
        recommendLevel = 1
      } else if (totalVramGB && totalVramGB >= limitation) {
        available = true
        comments = commentVRAMOccupy
        recommendLevel = -1
      } else if (totalVramGB && totalVramGB < limitation) {
        available = false
        comments = commentNotAvailable
        recommendLevel = -2
      } else {
        // ERROR?
      }

      const newResolution: Resolution = {
        ...resolution,
        comments,
        available,
        recommendLevel,
        calculateByPrecision,
        modelConfig: { ...sourceConfig, modelParameters },
        usingGPU: true,
      }
      resolutions.push(newResolution)
    }

    if (userSelectGPU) {
      const limitation = resolution.requirements.nf4
      let comments = commentVRAMFailed
      let available = true
      let recommendLevel = -1
      let calculateByPrecision: Precision = 'nf4'

      let modelParameters: ModelParameters = {
        modelName: resolution.modelName,
        device: 'WebGPU (Python)',
        precision: calculateByPrecision,
        storedLayers: 41,
        maxStoredLayers: 8,
      }
      if (availableVramGB && availableVramGB > limitation) {
        available = true
        comments = commentEasy
        recommendLevel = 1
      } else if (totalVramGB && totalVramGB >= limitation) {
        available = true
        comments = commentVRAMOccupy
        recommendLevel = -1
      } else if (totalVramGB && totalVramGB < limitation) {
        available = false
        comments = commentNotAvailable
        recommendLevel = -2
      } else {
        // ERROR?
      }

      const newResolution: Resolution = {
        ...resolution,
        comments,
        available,
        recommendLevel,
        calculateByPrecision,
        modelConfig: { ...sourceConfig, modelParameters },
        usingGPU: true,
      }
      resolutions.push(newResolution)
      continue
    }
  }

  // Find best resolution
  const bestResolutions: { [key in string]: Resolution } = {}
  for (let i = 0; i < resolutions.length; i++) {
    const resolution = resolutions[i]
    const recommendLevel = resolution.recommendLevel
    if (recommendLevel <= 0) continue
    const k =
      resolution.calculateByPrecision + (resolution.usingGPU ? 'gpu' : 'cpu')
    const currentBest = bestResolutions[k]
    if (currentBest) {
      if (
        resolution.requirements[resolution.calculateByPrecision] >
        currentBest.requirements[currentBest.calculateByPrecision]
      ) {
        bestResolutions[k] = resolution
      }
    } else {
      bestResolutions[k] = resolution
    }
  }

  Object.values(bestResolutions).forEach((resolution) => {
    resolution.recommendLevel = 2
    resolution.comments = commentRecommanded
  })

  resolutions.sort((a, b) => b.recommendLevel - a.recommendLevel)

  return resolutions
}

export type Resolution = {
  modelName: string
  /** in bytes */
  fileSize: number
  /** GB for precision */
  requirements: {
    [key in Precision]: number
  }
  // for sorting and auto selecion
  recommendLevel: RecommendLevel
  comments?: string
  available?: boolean
  modelConfig?: ModelConfig
  calculateByPrecision: Precision
  usingGPU: boolean
}
