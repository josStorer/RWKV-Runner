import { t } from 'i18next'
import { NavigateFunction } from 'react-router'
import { toast } from 'react-toastify'
import { FileExists, GetPyError } from '../../wailsjs/go/backend_golang/App'
import { WindowShow } from '../../wailsjs/runtime'
import cmdTaskChainStore from '../stores/cmdTaskChainStore'
import commonStore from '../stores/commonStore'
import { ModelConfig, Precision } from '../types/configs'
import {
  convertGGML,
  convertModel as convertModelFunc,
  convertSafetensors,
  convertSafetensorsWithPython,
} from '../utils/rwkv-task'
import { checkDependencies, getStrategy } from './index'

export const convertModel = async (
  selectedConfig: ModelConfig,
  navigate: NavigateFunction
) => {
  if (commonStore.platform === 'darwin') {
    toast(
      t(
        'MacOS is not yet supported for performing this operation, please do it manually.'
      ) + ' (backend-python/convert_model.py)',
      { type: 'info' }
    )
    return
  } else if (commonStore.platform === 'linux') {
    toast(
      t(
        'Linux is not yet supported for performing this operation, please do it manually.'
      ) + ' (backend-python/convert_model.py)',
      { type: 'info' }
    )
    return
  }

  const ok = await checkDependencies(navigate)
  if (!ok) return

  const modelPath = `${commonStore.settings.customModelsPath}/${selectedConfig.modelParameters.modelName}`
  if (await FileExists(modelPath)) {
    const strategy = getStrategy(selectedConfig)
    const newModelPath = modelPath + '-' + strategy.replace(/[:> *+]/g, '-')
    toast(t('Start Converting'), { autoClose: 2000, type: 'info' })
    const id = cmdTaskChainStore.newTaskChain('', [
      {
        name: t('转换模型')!,
        func: convertModelFunc,
        args: [
          commonStore.settings.customPythonPath,
          modelPath,
          strategy,
          newModelPath,
        ],
      },
    ])
    cmdTaskChainStore
      .startTaskChain(id)
      .then(async (success) => {
        if (!(await FileExists(newModelPath + '.pth'))) {
          toast(t('Convert Failed') + ' - ' + (await GetPyError()), {
            type: 'error',
          })
        } else {
          toast(`${t('Convert Success')} - ${newModelPath}`, {
            type: 'success',
          })
        }
      })
      .catch((e) => {
        const errMsg = e.message || e
        if (errMsg.includes('path contains space'))
          toast(
            `${t('Convert Failed')} - ${t('File Path Cannot Contain Space')}`,
            { type: 'error' }
          )
        else
          toast(`${t('Convert Failed')} - ${e.message || e}`, { type: 'error' })
      })
    setTimeout(WindowShow, 1000)
  } else {
    toast(`${t('Model Not Found')} - ${modelPath}`, { type: 'error' })
  }
}

export const convertToSt = async (
  selectedConfig: ModelConfig,
  navigate: NavigateFunction
) => {
  const webgpuPython =
    selectedConfig.modelParameters.device === 'WebGPU (Python)'
  if (webgpuPython) {
    const ok = await checkDependencies(navigate)
    if (!ok) return
  }

  const modelPath = `${commonStore.settings.customModelsPath}/${selectedConfig.modelParameters.modelName}`
  if (await FileExists(modelPath)) {
    toast(t('Start Converting'), { autoClose: 2000, type: 'info' })
    const newModelPath = modelPath.replace(/\.pth$/, '.st')
    const convert = webgpuPython
      ? (input: string, output: string) =>
          convertSafetensorsWithPython(
            commonStore.settings.customPythonPath,
            input,
            output
          )
      : convertSafetensors
    const id = cmdTaskChainStore.newTaskChain('', [
      {
        name: t('转换模型')!,
        func: convert,
        args: [modelPath, newModelPath],
      },
    ])
    cmdTaskChainStore
      .startTaskChain(id)
      .then(async (success) => {
        if (!(await FileExists(newModelPath))) {
          if (
            commonStore.platform === 'windows' ||
            commonStore.platform === 'linux'
          )
            toast(t('Convert Failed') + ' - ' + (await GetPyError()), {
              type: 'error',
            })
        } else {
          toast(`${t('Convert Success')} - ${newModelPath}`, {
            type: 'success',
          })
        }
      })
      .catch((e) => {
        const errMsg = e.message || e
        if (errMsg.includes('path contains space'))
          toast(
            `${t('Convert Failed')} - ${t('File Path Cannot Contain Space')}`,
            { type: 'error' }
          )
        else
          toast(`${t('Convert Failed')} - ${e.message || e}`, { type: 'error' })
      })
    setTimeout(WindowShow, 1000)
  } else {
    toast(`${t('Model Not Found')} - ${modelPath}`, { type: 'error' })
  }
}

export const convertToGGML = async (
  selectedConfig: ModelConfig,
  navigate: NavigateFunction
) => {
  const ok = await checkDependencies(navigate)
  if (!ok) return

  const modelPath = `${commonStore.settings.customModelsPath}/${selectedConfig.modelParameters.modelName}`
  if (await FileExists(modelPath)) {
    toast(t('Start Converting'), { autoClose: 2000, type: 'info' })
    const precision: Precision =
      selectedConfig.modelParameters.precision === 'Q5_1' ? 'Q5_1' : 'fp16'
    const newModelPath = modelPath.replace(/\.pth$/, `-${precision}.bin`)
    const id = cmdTaskChainStore.newTaskChain('', [
      {
        name: t('转换模型')!,
        func: convertGGML,
        args: [
          commonStore.settings.customPythonPath,
          modelPath,
          newModelPath,
          precision === 'Q5_1',
        ],
      },
    ])
    cmdTaskChainStore
      .startTaskChain(id)
      .then(async (success) => {
        if (!(await FileExists(newModelPath))) {
          if (
            commonStore.platform === 'windows' ||
            commonStore.platform === 'linux'
          )
            toast(t('Convert Failed') + ' - ' + (await GetPyError()), {
              type: 'error',
            })
        } else {
          toast(`${t('Convert Success')} - ${newModelPath}`, {
            type: 'success',
          })
        }
      })
      .catch((e) => {
        const errMsg = e.message || e
        if (errMsg.includes('path contains space'))
          toast(
            `${t('Convert Failed')} - ${t('File Path Cannot Contain Space')}`,
            { type: 'error' }
          )
        else
          toast(`${t('Convert Failed')} - ${e.message || e}`, { type: 'error' })
      })
    setTimeout(WindowShow, 1000)
  } else {
    toast(`${t('Model Not Found')} - ${modelPath}`, { type: 'error' })
  }
}
