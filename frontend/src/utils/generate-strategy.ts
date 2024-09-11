import commonStore from '../stores/commonStore'
import { ModelParameters, Precision } from '../types/configs'

const getStrategyForSwitchModelParameters = (param: ModelParameters) => {
  const {
    precision,
    device,
    useCustomCuda,
    customStrategy,
    tokenChunkSize,
    quantizedLayers,
    maxStoredLayers,
    storedLayers,
    modelName: modelNameParam,
  } = param
  const modelName = modelNameParam.toLowerCase()
  const avoidOverflow =
    precision !== 'fp32' &&
    modelName.includes('world') &&
    (modelName.includes('0.1b') ||
      modelName.includes('0.4b') ||
      modelName.includes('1.5b') ||
      modelName.includes('1b5'))
  let strategy = ''
  switch (device) {
    case 'CPU':
      if (avoidOverflow) strategy = 'cpu fp32 *1 -> '
      strategy += 'cpu '
      strategy += precision === 'int8' ? 'fp32i8' : 'fp32'
      break
    case 'WebGPU':
    case 'WebGPU (Python)':
      strategy +=
        precision === 'nf4'
          ? 'fp16i4'
          : precision === 'int8'
            ? 'fp16i8'
            : 'fp16'
      if (quantizedLayers) strategy += ` layer${quantizedLayers}`
      if (tokenChunkSize) strategy += ` chunk${tokenChunkSize}`
      break
    case 'CUDA':
    case 'CUDA-Beta':
      if (avoidOverflow)
        strategy = useCustomCuda ? 'cuda fp16 *1 -> ' : 'cuda fp32 *1 -> '
      strategy += 'cuda '
      strategy +=
        precision === 'int8' ? 'fp16i8' : precision === 'fp32' ? 'fp32' : 'fp16'
      if (storedLayers < maxStoredLayers) strategy += ` *${storedLayers}+`
      else strategy += ` -> cuda fp16 *1`
      break
    case 'MPS':
      if (avoidOverflow) strategy = 'mps fp32 *1 -> '
      strategy += 'mps '
      strategy += precision === 'int8' ? 'fp32i8' : 'fp32'
      break
    case 'Custom':
      strategy = customStrategy || ''
      break
  }
  return strategy
}

export type CompositionMode = 'MIDI' | 'ABC'

export type CPUOrGPU = 'GPU' | 'CPU'

export type Func = 'Chat' | 'Completion' | 'Composition' | 'Function Call'

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

// 需求：
// 1. 启动的时候依据剩余内存计算，然后如果依据总量计算， 理论上能运行更好的模型， 则会在启动时提示，资源占用，可运行XX模型
// 2. 选cpu的时候，用内存计算，依据剩余内存，计算当前启动用的配置，依据总内存，计算当前可用的理论最佳配置
// 3. 如果理论最佳比剩余内存算出来的更好，则提示用户
// 4. 用cuda fp16时，customCuda总是true
// 5. 显存满足这个，才用，其余都是WebGPU(nf4)
// 6. 精度只有nf4 fp16两种策略考虑,就是CUDA fp16 customCuda true 以及WebGPU(Python) nf4
// 7. tokenizer: undefined, deploy: false
// 8. ✅ switchModel
// 9. strategy刚才对应的就是cuda fp16和fp16i4
// 10. ✅ 硬编码模型名数组，然后模型名去commonStore.modelSourceList取信息，可以得到模型尺寸
// 11. 用户选CPU，那就只有rwkv.cpp with Q5_1， 如果是音乐模型，就是fp32
// 12. ✅ fp16载入的消耗显存量，以文件尺寸+1.5GB为值
// 13. ✅ int8以文件尺寸一半+1.5GB

// 关于显存占用推断问题
// 假设有当下的情况
// windows / ram 16gb, vram 8gb / ram 2gb available / vram 7gb available
// user choice: GPU, webgpu, need 4gb vram
// 根据奇奇的测试，模型要先过 ram 再到 vram
// 我的理解是 windows / macOS 上都有虚拟内存，所以，ram 2gb available 不会限制 4gb 的 model 的加载
// 但是到了 linux 上
// 经奇奇测试，加载 4gb 的模型时，如果 ram 2gb available，那么进程会被中断，导致模型无法加载
// 这个还影响我们的模型选择吗？

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
    gpuName,
    usedMemory: usedMemoryGB,
    totalMemory: totalMemoryGB,
    gpuUsage,
    usedVram: usedVramMB,
    totalVram: totalVramMB,
  } = commonStore.monitorData || {}

  console.log({
    gpuType,
    gpuName,
    usedMemory: usedMemoryGB,
    totalMemory: totalMemoryGB,
    gpuUsage,
    usedVramMB,
    totalVramMB,
  })

  const availableMemoryGB =
    totalMemoryGB !== undefined && usedMemoryGB !== undefined
      ? totalMemoryGB - usedMemoryGB
      : undefined

  const availableVramMB =
    totalVramMB !== undefined && usedVramMB !== undefined
      ? totalVramMB - usedVramMB
      : undefined

  let resolutions: Resolution[] = []

  // get
  let availableModels = func ? availableModelPairs[func] : []
  if (compositionMode && func === 'Composition') {
    availableModels = availableModels.filter((modelName) =>
      modelName.includes(compositionMode)
    )
  }
  console.log({ availableModels })

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
      modelName: modelName.replace('.pth', ''),
      fileSize: model!.size,
      available: true,
      recommendLevel: 0,
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

  for (let i = 0; i < resolutionSource.length; i++) {
    const resolution = resolutionSource[i]

    if (!userSelectGPU) {
      const limitation =
        func === 'Composition'
          ? resolution.requirements.fp32
          : resolution.requirements.Q5_1
      let comments = '获取内存状态错误'
      let available = true
      let recommendLevel = -1

      if (availableMemoryGB && availableMemoryGB > limitation) {
        available = true
        comments = '您的硬件可以轻松运行此模型'
        recommendLevel = 1
      } else if (totalMemoryGB && totalMemoryGB >= limitation) {
        available = true
        comments = '可以，资源占用，不建议使用，释放'
        recommendLevel = -1
      } else if (totalMemoryGB && totalMemoryGB < limitation) {
        available = false
        comments = '您的硬件无法运行此模型'
        recommendLevel = -2
      } else {
        // ERROR?
      }

      const newRes = { ...resolution, comments, available, recommendLevel }
      resolutions.push(newRes)
      continue
    }

    if (cudaAvailable && userSelectGPU) {
      const limitation = resolution.requirements.fp16

      let comments = '获取显存状态错误'
      let available = true
      let recommendLevel = -1

      if (availableMemoryGB && availableMemoryGB > limitation) {
        available = true
        comments = '您的硬件可以轻松运行此模型'
        recommendLevel = 1
      } else if (totalMemoryGB && totalMemoryGB >= limitation) {
        available = true
        comments = '可以，资源占用，不建议使用，释放'
        recommendLevel = -1
      } else if (totalMemoryGB && totalMemoryGB < limitation) {
        available = false
        comments = '您的硬件无法运行此模型'
        recommendLevel = -2
      } else {
        // ERROR?
      }

      const newRes = { ...resolution, comments, available, recommendLevel }
      resolutions.push(newRes)
      continue
    }

    if (userSelectGPU) {
      const limitation = resolution.requirements.nf4

      let comments = '获取显存状态错误'
      let available = true
      let recommendLevel = -1

      if (availableMemoryGB && availableMemoryGB > limitation) {
        available = true
        comments = '您的硬件可以轻松运行此模型'
        recommendLevel = 1
      } else if (totalMemoryGB && totalMemoryGB >= limitation) {
        available = true
        comments = '可以，资源占用，不建议使用，释放'
        recommendLevel = -1
      } else if (totalMemoryGB && totalMemoryGB < limitation) {
        available = false
        comments = '您的硬件无法运行此模型'
        recommendLevel = -2
      } else {
        // ERROR?
      }

      const newRes = { ...resolution, comments, available, recommendLevel }
      resolutions.push(newRes)
      continue
    }
  }

  resolutions.sort((a, b) => b.recommendLevel - a.recommendLevel)

  return resolutions
}

export type Resolution = {
  modelName: string
  fileSize: number
  requirements: {
    [key in Precision]: number
  }
  // for sorting and auto selecion
  recommendLevel: number
  // 注解，在开发阶段先不用格式化、本地化，等着需求进一步确定和明确了再说
  comments?: string
  available?: boolean
  modelParameters?: ModelParameters
  calculateByPrecision?: Precision
}
