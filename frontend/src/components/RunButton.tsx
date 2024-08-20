import React, { FC, MouseEventHandler, ReactElement } from 'react'
import { Button } from '@fluentui/react-components'
import { Play16Regular, Stop16Regular } from '@fluentui/react-icons'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router'
import { toast } from 'react-toastify'
import {
  AddToDownloadList,
  FileExists,
  IsPortAvailable,
  StartServer,
  StartWebGPUServer,
} from '../../wailsjs/go/backend_golang/App'
import { WindowShow } from '../../wailsjs/runtime'
import { exit, getStatus, readRoot, switchModel, updateConfig } from '../apis'
import {
  defaultCompositionABCPrompt,
  defaultCompositionPrompt,
} from '../pages/defaultConfigs'
import commonStore, { ModelStatus } from '../stores/commonStore'
import { Precision } from '../types/configs'
import {
  checkDependencies,
  getHfDownloadUrl,
  getStrategy,
  toastWithButton,
} from '../utils'
import { convertToGGML, convertToSt } from '../utils/convert-model'
import { ToolTipButton } from './ToolTipButton'

const mainButtonText = {
  [ModelStatus.Offline]: 'Run',
  [ModelStatus.Starting]: 'Starting',
  [ModelStatus.Loading]: 'Loading',
  [ModelStatus.Working]: 'Stop',
}

const iconModeButtonIcon: { [modelStatus: number]: ReactElement } = {
  [ModelStatus.Offline]: <Play16Regular />,
  [ModelStatus.Starting]: <Stop16Regular />,
  [ModelStatus.Loading]: <Stop16Regular />,
  [ModelStatus.Working]: <Stop16Regular />,
}

export const RunButton: FC<{
  onClickRun?: MouseEventHandler
  iconMode?: boolean
}> = observer(({ onClickRun, iconMode }) => {
  const { t } = useTranslation()
  const navigate = useNavigate()

  const onClickMainButton = async () => {
    if (commonStore.status.status === ModelStatus.Offline) {
      commonStore.setStatus({ status: ModelStatus.Starting })

      const modelConfig = commonStore.getCurrentModelConfig()
      const webgpu = modelConfig.modelParameters.device === 'WebGPU'
      const webgpuPython =
        modelConfig.modelParameters.device === 'WebGPU (Python)'
      const cpp = modelConfig.modelParameters.device === 'CPU (rwkv.cpp)'
      let modelName = ''
      let modelPath = ''
      if (modelConfig && modelConfig.modelParameters) {
        modelName = modelConfig.modelParameters.modelName
        modelPath = `${commonStore.settings.customModelsPath}/${modelName}`
      } else {
        toast(t('Model Config Exception'), { type: 'error' })
        commonStore.setStatus({ status: ModelStatus.Offline })
        return
      }

      const currentModelSource = commonStore.modelSourceList.find(
        (item) => item.name === modelName
      )

      const showDownloadPrompt = (promptInfo: string, downloadName: string) => {
        toastWithButton(promptInfo, t('Download'), () => {
          const downloadUrl = currentModelSource?.downloadUrl
          if (downloadUrl) {
            toastWithButton(
              `${t('Downloading')} ${downloadName}`,
              t('Check'),
              () => {
                navigate({ pathname: '/downloads' })
              },
              { autoClose: 3000 }
            )
            AddToDownloadList(modelPath, getHfDownloadUrl(downloadUrl))
          } else {
            toast(t('Can not find download url'), { type: 'error' })
          }
        })
      }

      if (webgpu || webgpuPython) {
        if (!['.st', '.safetensors'].some((ext) => modelPath.endsWith(ext))) {
          const stModelPath = modelPath.replace(/\.pth$/, '.st')
          if (await FileExists(stModelPath)) {
            modelPath = stModelPath
          } else if (!(await FileExists(modelPath))) {
            showDownloadPrompt(t('Model file not found'), modelName)
            commonStore.setStatus({ status: ModelStatus.Offline })
            return
          } else if (!currentModelSource?.isComplete) {
            showDownloadPrompt(
              t('Model file download is not complete'),
              modelName
            )
            commonStore.setStatus({ status: ModelStatus.Offline })
            return
          } else {
            toastWithButton(
              t('Please convert model to safe tensors format first'),
              t('Convert'),
              () => {
                convertToSt(modelConfig, navigate)
              }
            )
            commonStore.setStatus({ status: ModelStatus.Offline })
            return
          }
        }
      }

      if (!webgpu && !webgpuPython) {
        if (['.st', '.safetensors'].some((ext) => modelPath.endsWith(ext))) {
          toast(
            t('Please change Strategy to WebGPU to use safetensors format'),
            { type: 'error' }
          )
          commonStore.setStatus({ status: ModelStatus.Offline })
          return
        }
      }

      if (!webgpu) {
        const ok = await checkDependencies(navigate)
        if (!ok) return
      }

      if (cpp) {
        if (!['.bin'].some((ext) => modelPath.endsWith(ext))) {
          const precision: Precision =
            modelConfig.modelParameters.precision === 'Q5_1' ? 'Q5_1' : 'fp16'
          const ggmlModelPath = modelPath.replace(/\.pth$/, `-${precision}.bin`)
          if (await FileExists(ggmlModelPath)) {
            modelPath = ggmlModelPath
          } else if (!(await FileExists(modelPath))) {
            showDownloadPrompt(t('Model file not found'), modelName)
            commonStore.setStatus({ status: ModelStatus.Offline })
            return
          } else if (!currentModelSource?.isComplete) {
            showDownloadPrompt(
              t('Model file download is not complete'),
              modelName
            )
            commonStore.setStatus({ status: ModelStatus.Offline })
            return
          } else {
            toastWithButton(
              t('Please convert model to GGML format first'),
              t('Convert'),
              () => {
                convertToGGML(modelConfig, navigate)
              }
            )
            commonStore.setStatus({ status: ModelStatus.Offline })
            return
          }
        }
      }

      if (!cpp) {
        if (['.bin'].some((ext) => modelPath.endsWith(ext))) {
          toast(
            t('Please change Strategy to CPU (rwkv.cpp) to use ggml format'),
            { type: 'error' }
          )
          commonStore.setStatus({ status: ModelStatus.Offline })
          return
        }
      }

      if (!(await FileExists(modelPath))) {
        showDownloadPrompt(t('Model file not found'), modelName)
        commonStore.setStatus({ status: ModelStatus.Offline })
        return
      } // If the user selects the .pth model with WebGPU mode, modelPath will be set to the .st model.
      // However, if the .pth model is deleted, modelPath will exist and isComplete will be false.
      else if (!currentModelSource?.isComplete && modelPath.endsWith('.pth')) {
        showDownloadPrompt(t('Model file download is not complete'), modelName)
        commonStore.setStatus({ status: ModelStatus.Offline })
        return
      }

      const port = modelConfig.apiParameters.apiPort

      if (!(await IsPortAvailable(port))) {
        await exit(1000).catch(() => {})
        if (!(await IsPortAvailable(port))) {
          toast(
            t(
              'Port is occupied. Change it in Configs page or close the program that occupies the port.'
            ),
            { type: 'error' }
          )
          commonStore.setStatus({ status: ModelStatus.Offline })
          return
        }
      }

      const startServer = webgpu
        ? (_: string, port: number, host: string) =>
            StartWebGPUServer(port, host)
        : StartServer
      const isUsingCudaBeta = modelConfig.modelParameters.device === 'CUDA-Beta'

      startServer(
        commonStore.settings.customPythonPath,
        port,
        commonStore.settings.host !== '127.0.0.1' ? '0.0.0.0' : '127.0.0.1',
        !!modelConfig.enableWebUI,
        isUsingCudaBeta,
        cpp,
        webgpuPython
      ).catch((e) => {
        const errMsg = e.message || e
        if (errMsg.includes('path contains space'))
          toast(`${t('Error')} - ${t('File Path Cannot Contain Space')}`, {
            type: 'error',
          })
        else toast(t('Error') + ' - ' + errMsg, { type: 'error' })
      })
      setTimeout(WindowShow, 1000)
      setTimeout(WindowShow, 2000)
      setTimeout(WindowShow, 3000)

      let timeoutCount = 6
      let loading = false
      const intervalId = setInterval(() => {
        readRoot()
          .then(async (r) => {
            if (r.ok && !loading) {
              loading = true
              clearInterval(intervalId)
              if (!webgpu) {
                await getStatus().then((status) => {
                  if (status) commonStore.setStatus(status)
                })
              }
              commonStore.setStatus({ status: ModelStatus.Loading })
              const loadingId = toast(t('Loading Model'), {
                type: 'info',
                autoClose: false,
              })
              if (!webgpu) {
                updateConfig(t, {
                  max_tokens: modelConfig.apiParameters.maxResponseToken,
                  temperature: modelConfig.apiParameters.temperature,
                  top_p: modelConfig.apiParameters.topP,
                  presence_penalty: modelConfig.apiParameters.presencePenalty,
                  frequency_penalty: modelConfig.apiParameters.frequencyPenalty,
                  penalty_decay: modelConfig.apiParameters.penaltyDecay,
                  global_penalty: modelConfig.apiParameters.globalPenalty,
                  state: modelConfig.apiParameters.stateModel,
                }).then(async (r) => {
                  if (r.status !== 200) {
                    const error = await r.text()
                    if (error.includes('state shape mismatch'))
                      toast(t('State model mismatch'), { type: 'error' })
                    else if (
                      error.includes(
                        'file format of the model or state model not supported'
                      )
                    )
                      toast(
                        t(
                          'File format of the model or state model not supported'
                        ),
                        { type: 'error' }
                      )
                    else toast(error, { type: 'error' })
                  }
                })
              }

              const strategy = getStrategy(modelConfig)
              let customCudaFile = ''
              if (
                (modelConfig.modelParameters.device.startsWith('CUDA') ||
                  modelConfig.modelParameters.device === 'Custom') &&
                modelConfig.modelParameters.useCustomCuda &&
                !strategy
                  .split('->')
                  .some((s) => ['cuda', 'fp32'].every((v) => s.includes(v)))
              ) {
                if (commonStore.platform === 'windows') {
                  // this part is currently unused because there's no longer a need to use different kernels for different GPUs, but it might still be needed in the future
                  //
                  // customCudaFile = getSupportedCustomCudaFile(isUsingCudaBeta);
                  // if (customCudaFile) {
                  //   let kernelTargetPath: string;
                  //   if (isUsingCudaBeta)
                  //     kernelTargetPath = './backend-python/rwkv_pip/beta/wkv_cuda.pyd';
                  //   else
                  //     kernelTargetPath = './backend-python/rwkv_pip/wkv_cuda.pyd';
                  //   await CopyFile(customCudaFile, kernelTargetPath).catch(() => {
                  //     FileExists(kernelTargetPath).then((exist) => {
                  //       if (!exist) {
                  //         customCudaFile = '';
                  //         toast(t('Failed to copy custom cuda file'), { type: 'error' });
                  //       }
                  //     });
                  //   });
                  // } else
                  //   toast(t('Supported custom cuda file not found'), { type: 'warning' });
                  customCudaFile = 'any'
                } else {
                  customCudaFile = 'any'
                }
              }

              switchModel({
                model: modelPath,
                strategy: strategy,
                tokenizer: modelConfig.modelParameters.useCustomTokenizer
                  ? modelConfig.modelParameters.customTokenizer
                  : undefined,
                customCuda: customCudaFile !== '',
                deploy: modelConfig.enableWebUI,
              })
                .then(async (r) => {
                  if (r.ok) {
                    commonStore.setStatus({ status: ModelStatus.Working })
                    let buttonNameMap = {
                      novel: 'Completion',
                      abc: 'Composition',
                      midi: 'Composition',
                    }
                    let buttonName = 'Chat'
                    buttonName =
                      Object.entries(buttonNameMap).find(([key, value]) =>
                        modelName.toLowerCase().includes(key)
                      )?.[1] || buttonName
                    const buttonFn = () => {
                      navigate({ pathname: '/' + buttonName.toLowerCase() })
                    }
                    if (
                      modelName.toLowerCase().includes('abc') &&
                      commonStore.compositionParams.prompt ===
                        defaultCompositionPrompt
                    ) {
                      commonStore.setCompositionParams({
                        ...commonStore.compositionParams,
                        prompt: defaultCompositionABCPrompt,
                      })
                      commonStore.setCompositionSubmittedPrompt(
                        defaultCompositionABCPrompt
                      )
                    }

                    if (
                      modelConfig.modelParameters.device.startsWith('CUDA') &&
                      modelConfig.modelParameters.storedLayers <
                        modelConfig.modelParameters.maxStoredLayers &&
                      commonStore.monitorData &&
                      commonStore.monitorData.totalVram !== 0 &&
                      commonStore.monitorData.usedVram /
                        commonStore.monitorData.totalVram <
                        0.9
                    )
                      toast(
                        t(
                          'You can increase the number of stored layers in Configs page to improve performance'
                        ),
                        { type: 'info' }
                      )
                    toastWithButton(
                      t('Startup Completed'),
                      t(buttonName),
                      buttonFn,
                      { type: 'success', autoClose: 3000 }
                    )
                  } else if (r.status === 304) {
                    toast(t('Loading Model'), { type: 'info' })
                  } else {
                    commonStore.setStatus({ status: ModelStatus.Offline })
                    const error = await r.text()
                    const errorsMap = {
                      'not enough memory':
                        'Memory is not enough, try to increase the virtual memory or use a smaller model.',
                      'not compiled with CUDA':
                        'Bad PyTorch version, please reinstall PyTorch with cuda.',
                      'invalid header or archive is corrupted':
                        'The model file is corrupted, please download again.',
                      'no NVIDIA driver':
                        "Found no NVIDIA driver, please install the latest driver. If you are not using an Nvidia GPU, please switch the 'Strategy' to WebGPU or CPU in the Configs page.",
                      'CUDA out of memory':
                        'VRAM is not enough, please reduce stored layers or use a lower precision in Configs page.',
                      'Ninja is required to load C++ extensions':
                        'Failed to enable custom CUDA kernel, ninja is required to load C++ extensions. You may be using the CPU version of PyTorch, please reinstall PyTorch with CUDA. Or if you are using a custom Python interpreter, you must compile the CUDA kernel by yourself or disable Custom CUDA kernel acceleration.',
                      're-convert the model':
                        'Model has been converted and does not match current strategy. If you are using a new strategy, re-convert the model.',
                    }
                    const matchedError = Object.entries(errorsMap).find(
                      ([key, _]) => error.includes(key)
                    )
                    const message = matchedError ? t(matchedError[1]) : error
                    toast(t('Failed to switch model') + ' - ' + message, {
                      autoClose: 5000,
                      type: 'error',
                    })
                  }
                })
                .catch((e) => {
                  commonStore.setStatus({ status: ModelStatus.Offline })
                  toast(
                    t('Failed to switch model') + ' - ' + (e.message || e),
                    { type: 'error' }
                  )
                })
                .finally(() => {
                  toast.dismiss(loadingId)
                })
            }
          })
          .catch(() => {
            if (timeoutCount <= 0) {
              clearInterval(intervalId)
              commonStore.setStatus({ status: ModelStatus.Offline })
            }
          })

        timeoutCount--
      }, 1000)
    } else {
      commonStore.setStatus({ status: ModelStatus.Offline })
      exit().then((r) => {
        if (r.status === 403)
          if (commonStore.platform !== 'linux')
            toast(
              t(
                'Server is working on deployment mode, please close the terminal window manually'
              ),
              { type: 'info' }
            )
          else
            toast(
              t(
                'Server is working on deployment mode, please exit the program manually to stop the server'
              ),
              { type: 'info' }
            )
      })
    }
  }

  const onClick = async (e: any) => {
    if (commonStore.status.status === ModelStatus.Offline) await onClickRun?.(e)
    await onClickMainButton()
  }

  return iconMode ? (
    <ToolTipButton
      disabled={commonStore.status.status === ModelStatus.Starting}
      icon={iconModeButtonIcon[commonStore.status.status]}
      desc={t(mainButtonText[commonStore.status.status])}
      size="small"
      onClick={onClick}
    />
  ) : (
    <Button
      disabled={commonStore.status.status === ModelStatus.Starting}
      appearance="primary"
      size="large"
      onClick={onClick}
    >
      {t(mainButtonText[commonStore.status.status])}
    </Button>
  )
})
