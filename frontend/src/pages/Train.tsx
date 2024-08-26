import React, { FC, useEffect, useRef, useState } from 'react'
import {
  Button,
  Dropdown,
  Input,
  Option,
  Select,
  Switch,
  Tab,
  TabList,
} from '@fluentui/react-components'
import {
  DataUsageSettings20Regular,
  Folder20Regular,
} from '@fluentui/react-icons'
import { SelectTabEventHandler } from '@fluentui/react-tabs'
import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
} from 'chart.js'
import { t } from 'i18next'
import { observer } from 'mobx-react-lite'
import { Line } from 'react-chartjs-2'
import { ChartJSOrUndefined } from 'react-chartjs-2/dist/types'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router'
import { toast } from 'react-toastify'
import {
  ConvertData,
  FileExists,
  GetPyError,
  MergeLora,
  OpenFileFolder,
  WslCommand,
  WslEnable,
  WslInstallUbuntu,
  WslIsEnabled,
  WslStart,
  WslStop,
} from '../../wailsjs/go/backend_golang/App'
import { WindowShow } from '../../wailsjs/runtime'
import { DialogButton } from '../components/DialogButton'
import { Labeled } from '../components/Labeled'
import { Section } from '../components/Section'
import { ToolTipButton } from '../components/ToolTipButton'
import commonStore from '../stores/commonStore'
import {
  DataProcessParameters,
  LoraFinetuneParameters,
  LoraFinetunePrecision,
  TrainNavigationItem,
} from '../types/train'
import { checkDependencies, toastWithButton } from '../utils'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Title,
  Legend
)

const parseLossData = (data: string) => {
  const regex =
    /Epoch (\d+):\s+(\d+%)\|[\s\S]*\| (\d+)\/(\d+) \[(\S+)<(\S+),\s+(\S+), loss=(\S+),[\s\S]*\]/g
  const matches = Array.from(data.matchAll(regex))
  if (matches.length === 0) return false
  const lastMatch = matches[matches.length - 1]
  const epoch = parseInt(lastMatch[1])
  const loss = parseFloat(lastMatch[8])
  commonStore.setChartTitle(
    `Epoch ${epoch}: ${lastMatch[2]} - ${lastMatch[3]}/${lastMatch[4]} - ${lastMatch[5]}/${lastMatch[6]} - ${lastMatch[7]} Loss=${loss}`
  )
  addLossDataToChart(epoch, loss)
  if (loss > 5)
    toast(
      t(
        'Loss is too high, please check the training data, and ensure your gpu driver is up to date.'
      ),
      {
        type: 'warning',
        toastId: 'train_loss_high',
      }
    )
  return true
}

let chartLine: ChartJSOrUndefined<'line', (number | null)[], string>

const addLossDataToChart = (epoch: number, loss: number) => {
  const epochIndex = commonStore.chartData.labels!.findIndex((l) =>
    l.includes(epoch.toString())
  )
  if (epochIndex === -1) {
    if (epoch === 0) {
      commonStore.chartData.labels!.push('Init')
      commonStore.chartData.datasets[0].data = [
        ...commonStore.chartData.datasets[0].data,
        loss,
      ]
    }
    commonStore.chartData.labels!.push('Epoch ' + epoch.toString())
    commonStore.chartData.datasets[0].data = [
      ...commonStore.chartData.datasets[0].data,
      loss,
    ]
  } else {
    if (chartLine) {
      const newData = [...commonStore.chartData.datasets[0].data]
      newData[epochIndex] = loss
      chartLine.data.datasets[0].data = newData
      chartLine.update()
    }
  }
  commonStore.setChartData(commonStore.chartData)
}

const loraFinetuneParametersOptions: Array<
  [key: keyof LoraFinetuneParameters, type: string, name: string]
> = [
  ['devices', 'number', 'Devices'],
  ['precision', 'LoraFinetunePrecision', 'Precision'],
  ['gradCp', 'boolean', 'Gradient Checkpoint'],
  ['ctxLen', 'number', 'Context Length'],
  ['epochSteps', 'number', 'Epoch Steps'],
  ['epochCount', 'number', 'Epoch Count'],
  ['epochBegin', 'number', 'Epoch Begin'],
  ['epochSave', 'number', 'Epoch Save'],
  ['lrInit', 'string', 'Learning Rate Init'],
  ['lrFinal', 'string', 'Learning Rate Final'],
  ['microBsz', 'number', 'Micro Batch Size'],
  ['accumGradBatches', 'number', 'Accumulate Gradient Batches'],
  ['warmupSteps', 'number', 'Warmup Steps'],
  ['adamEps', 'string', 'Adam Epsilon'],
  ['beta1', 'number', 'Beta 1'],
  ['beta2', 'number', 'Beta 2'],
  ['loraR', 'number', 'LoRA R'],
  ['loraAlpha', 'number', 'LoRA Alpha'],
  ['loraDropout', 'number', 'LoRA Dropout'],
  ['beta1', 'any', ''],
  // ['preFfn', 'boolean', 'Pre-FFN'],
  // ['headQk', 'boolean', 'Head QK']
]

const showError = (e: any) => {
  const msg = e.message || e
  if (msg === 'wsl not running') {
    toast(
      t(
        'WSL is not running, please retry. If it keeps happening, it means you may be using an outdated version of WSL, run "wsl --update" to update.'
      ),
      { type: 'error' }
    )
  } else {
    toast(t(msg), { type: 'error', toastId: 'train_error' })
  }
}

// error key should be lowercase
const errorsMap = Object.entries({
  ['python3 ./finetune/lora/$modelInfo'.toLowerCase()]:
    'Memory is not enough, try to increase the virtual memory (Swap of WSL) or use a smaller base model.',
  'cuda out of memory': 'VRAM is not enough',
  'valueerror: high <= 0':
    'Training data is not enough, reduce context length or add more data for training',
  "+= '+ptx'":
    'Can not find an Nvidia GPU. Perhaps the gpu driver of windows is too old, or you are using WSL 1 for training, please upgrade to WSL 2. e.g. Run "wsl --set-version Ubuntu-22.04 2"',
  'size mismatch for blocks':
    'Size mismatch for blocks. You are attempting to continue training from the LoRA model, but it does not match the base model. Please set LoRA model to None.',
  'cuda_home environment variable is not set': 'Matched CUDA is not installed',
  'unsupported gpu architecture': 'Matched CUDA is not installed',
  "error building extension 'fused_adam'": 'Matched CUDA is not installed',
  'rwkv{version} is not supported':
    'This version of RWKV is not supported yet.',
  'no such file':
    'Failed to find the base model, please try to change your base model.',
  'modelinfo is invalid':
    'Failed to load model, try to increase the virtual memory (Swap of WSL) or use a smaller base model.',
})

export const wslHandler = (data: string) => {
  if (data) {
    addWslMessage(data)
    const ok = parseLossData(data)
    if (!ok)
      for (const [key, value] of errorsMap) {
        if (data.toLowerCase().includes(key)) {
          showError(value)
          return
        }
      }
  }
}

const addWslMessage = (message: string) => {
  const newData = commonStore.wslStdout + '\n' + message
  let lines = newData.split('\n')
  const result = lines.slice(-100).join('\n')
  commonStore.setWslStdout(result)
}

const TerminalDisplay: FC = observer(() => {
  const bodyRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    if (bodyRef.current)
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }

  useEffect(() => {
    scrollToBottom()
  })

  return (
    <div
      ref={bodyRef}
      className="grow overflow-y-auto overflow-x-hidden rounded-md border-2 border-gray-500"
    >
      <div className="whitespace-pre-line">{commonStore.wslStdout}</div>
    </div>
  )
})

const Terminal: FC = observer(() => {
  const { t } = useTranslation()
  const [input, setInput] = useState('')

  const handleKeyDown = (e: any) => {
    e.stopPropagation()
    if (e.keyCode === 13) {
      e.preventDefault()
      if (!input) return

      WslStart()
        .then(() => {
          addWslMessage('WSL> ' + input)
          setInput('')
          WslCommand(input).then(WindowShow).catch(showError)
        })
        .catch(showError)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <TerminalDisplay />
      <div className="flex items-center gap-2">
        WSL:
        <Input
          className="grow"
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
          }}
          onKeyDown={handleKeyDown}
        ></Input>
        <Button
          onClick={() => {
            WslStop()
              .then(() => {
                toast(t('Command Stopped'), { type: 'success' })
              })
              .catch(showError)
          }}
        >
          {t('Stop')}
        </Button>
      </div>
    </div>
  )
})

const LoraFinetune: FC = observer(() => {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const chartRef =
    useRef<ChartJSOrUndefined<'line', (number | null)[], string>>(null)

  const dataParams = commonStore.dataProcessParams
  const loraParams = commonStore.loraFinetuneParams

  if (chartRef.current) chartLine = chartRef.current

  const setDataParams = (newParams: Partial<DataProcessParameters>) => {
    commonStore.setDataProcessParams({
      ...dataParams,
      ...newParams,
    })
  }

  const setLoraParams = (newParams: Partial<LoraFinetuneParameters>) => {
    commonStore.setLoraFinetuneParameters({
      ...loraParams,
      ...newParams,
    })
  }

  useEffect(() => {
    if (loraParams.baseModel === '')
      setLoraParams({
        baseModel:
          commonStore.modelSourceList.find((m) => m.isComplete)?.name || '',
      })
  }, [])

  const StartLoraFinetune = async () => {
    const ok = await checkDependencies(navigate)
    if (!ok) return

    const convertedDataPath =
      './finetune/json2binidx_tool/data/' +
      dataParams.dataPath
        .replace(/[\/\\]$/, '')
        .split(/[\/\\]/)
        .pop()!
        .split('.')[0] +
      '_text_document'
    if (!(await FileExists(convertedDataPath + '.idx'))) {
      toast(t('Please convert data first.'), { type: 'error' })
      return
    }

    WslIsEnabled()
      .then(() => {
        WslStart()
          .then(() => {
            setTimeout(WindowShow, 1000)

            let ctxLen = loraParams.ctxLen
            if (dataParams.dataPath === 'finetune/data/sample.jsonl') {
              ctxLen = 150
              toast(
                t(
                  'You are using sample data for training. For formal training, please make sure to create your own jsonl file.'
                ),
                {
                  type: 'info',
                  autoClose: 6000,
                }
              )
            }

            commonStore.setChartData({
              labels: [],
              datasets: [
                {
                  label: 'Loss',
                  data: [],
                  borderColor: 'rgb(53, 162, 235)',
                  backgroundColor: 'rgba(53, 162, 235, 0.5)',
                },
              ],
            })
            WslCommand(
              `export cnMirror=${commonStore.settings.cnMirror ? '1' : '0'} ` +
                `&& export loadModel=models/${loraParams.baseModel} ` +
                `&& sed -i 's/\\r$//' finetune/install-wsl-dep-and-train.sh ` +
                `&& chmod +x finetune/install-wsl-dep-and-train.sh && ./finetune/install-wsl-dep-and-train.sh ` +
                (loraParams.baseModel
                  ? `--load_model models/${loraParams.baseModel} `
                  : '') +
                (loraParams.loraLoad
                  ? `--lora_load lora-models/${loraParams.loraLoad} `
                  : '') +
                `--data_file ${convertedDataPath} ` +
                `--ctx_len ${ctxLen} --epoch_steps ${loraParams.epochSteps} --epoch_count ${loraParams.epochCount} ` +
                `--epoch_begin ${loraParams.epochBegin} --epoch_save ${loraParams.epochSave} ` +
                `--micro_bsz ${loraParams.microBsz} --accumulate_grad_batches ${loraParams.accumGradBatches} ` +
                `--pre_ffn ${loraParams.preFfn ? '0' : '0'} --head_qk ${loraParams.headQk ? '0' : '0'} --lr_init ${loraParams.lrInit} --lr_final ${loraParams.lrFinal} ` +
                `--warmup_steps ${loraParams.warmupSteps} ` +
                `--beta1 ${loraParams.beta1} --beta2 ${loraParams.beta2} --adam_eps ${loraParams.adamEps} ` +
                `--devices ${loraParams.devices} --precision ${loraParams.precision} ` +
                `--grad_cp ${loraParams.gradCp ? '1' : '0'} ` +
                `--lora_r ${loraParams.loraR} --lora_alpha ${loraParams.loraAlpha} --lora_dropout ${loraParams.loraDropout}`
            ).catch(showError)
          })
          .catch((e) => {
            WindowShow()
            const msg = e.message || e
            if (
              msg === 'ubuntu not found' ||
              msg.includes('could not obtain registered distros')
            ) {
              toastWithButton(
                t('Ubuntu is not installed, do you want to install it?'),
                t('Install Ubuntu'),
                () => {
                  WslInstallUbuntu()
                    .then(() => {
                      WindowShow()
                      toast(
                        t(
                          'Please install Ubuntu using Microsoft Store, after installation click the Open button in Microsoft Store and then click the Train button'
                        ),
                        {
                          type: 'info',
                          autoClose: 10000,
                        }
                      )
                    })
                    .catch(showError)
                }
              )
            } else {
              showError(msg)
            }
          })
      })
      .catch((e) => {
        const msg = e.message || e

        const enableWsl = (forceMode: boolean) => {
          WindowShow()
          toastWithButton(
            t('WSL is not enabled, do you want to enable it?'),
            t('Enable WSL'),
            () => {
              WslEnable(forceMode)
                .then(() => {
                  WindowShow()
                  toast(
                    t(
                      'After installation, please restart your computer to enable WSL'
                    ),
                    {
                      type: 'info',
                      autoClose: false,
                    }
                  )
                })
                .catch(showError)
            }
          )
        }

        if (msg === 'wsl is not enabled') {
          enableWsl(true)
        } else if (msg.includes('wsl.state: The system cannot find the file')) {
          enableWsl(true)
        } else {
          showError(msg)
        }
      })
  }

  return (
    <div className="flex h-full w-full flex-col gap-2">
      {(commonStore.wslStdout.length > 0 ||
        commonStore.chartData.labels!.length !== 0) && (
        <div className="flex" style={{ height: '35%' }}>
          {commonStore.wslStdout.length > 0 &&
            commonStore.chartData.labels!.length === 0 && <TerminalDisplay />}
          {commonStore.chartData.labels!.length !== 0 && (
            <Line
              ref={chartRef}
              data={commonStore.chartData}
              options={{
                responsive: true,
                showLine: true,
                plugins: {
                  legend: {
                    position: 'right',
                    align: 'start',
                  },
                  title: {
                    display: true,
                    text: commonStore.chartTitle,
                  },
                },
                scales: {
                  y: {
                    beginAtZero: true,
                  },
                },
                maintainAspectRatio: false,
              }}
              style={{ width: '100%' }}
            />
          )}
        </div>
      )}
      <div>
        <Section
          title={t('Data Process')}
          content={
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                {t('Data Path')}
                <Input
                  className="grow"
                  style={{ minWidth: 0 }}
                  value={dataParams.dataPath}
                  onChange={(e, data) => {
                    setDataParams({ dataPath: data.value })
                  }}
                />
                <DialogButton
                  text={t('Help')}
                  title={t('Help')}
                  markdown
                  content={t(
                    'The data path should be a directory or a file in jsonl format (more formats will be supported in the future).\n\n' +
                      'When you provide a directory path, all the txt files within that directory will be automatically converted into training data. ' +
                      'This is commonly used for large-scale training in writing, code generation, or knowledge bases.\n\n' +
                      'The jsonl format file can be referenced at https://github.com/josStorer/RWKV-Runner/blob/master/finetune/data/sample.jsonl.\n' +
                      "You can also write it similar to OpenAI's playground format, as shown in https://platform.openai.com/playground/p/default-chat.\n" +
                      'Even for multi-turn conversations, they must be written in a single line using `\\n` to indicate line breaks. ' +
                      'If they are different dialogues or topics, they should be written in separate lines.'
                  )}
                />
                <ToolTipButton
                  desc={t('Open Folder')}
                  icon={<Folder20Regular />}
                  onClick={() => {
                    OpenFileFolder(dataParams.dataPath)
                  }}
                />
              </div>
              <div className="flex items-center gap-2">
                {t('Vocab Path')}
                <Input
                  className="grow"
                  style={{ minWidth: 0 }}
                  value={dataParams.vocabPath}
                  onChange={(e, data) => {
                    setDataParams({ vocabPath: data.value })
                  }}
                />
                <Button
                  appearance="secondary"
                  onClick={async () => {
                    const ok = await checkDependencies(navigate)
                    if (!ok) return
                    const outputPrefix =
                      './finetune/json2binidx_tool/data/' +
                      dataParams.dataPath
                        .replace(/[\/\\]$/, '')
                        .split(/[\/\\]/)
                        .pop()!
                        .split('.')[0]
                    ConvertData(
                      commonStore.settings.customPythonPath,
                      dataParams.dataPath.replaceAll('\\', '/'),
                      outputPrefix,
                      dataParams.vocabPath
                    )
                      .then(async () => {
                        if (
                          !(await FileExists(
                            outputPrefix + '_text_document.idx'
                          ))
                        ) {
                          if (
                            commonStore.platform === 'windows' ||
                            commonStore.platform === 'linux'
                          )
                            toast(
                              t('Failed to convert data') +
                                ' - ' +
                                (await GetPyError()),
                              { type: 'error' }
                            )
                        } else {
                          toast(t('Convert Data successfully'), {
                            type: 'success',
                          })
                        }
                      })
                      .catch(showError)
                  }}
                >
                  {t('Convert')}
                </Button>
              </div>
            </div>
          }
        />
      </div>
      <Section
        title={t('Train Parameters')}
        content={
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="flex items-center gap-2">
              <div className="shrink-0">{t('Base Model')}</div>
              <Select
                style={{ minWidth: 0 }}
                className="grow"
                value={loraParams.baseModel}
                onChange={(e, data) => {
                  setLoraParams({
                    baseModel: data.value,
                  })
                }}
              >
                {commonStore.modelSourceList.map(
                  (modelItem, index) =>
                    modelItem.isComplete && (
                      <option key={index} value={modelItem.name}>
                        {modelItem.name}
                      </option>
                    )
                )}
              </Select>
              <ToolTipButton
                desc={t('Manage Models')}
                icon={<DataUsageSettings20Regular />}
                onClick={() => {
                  navigate({ pathname: '/models' })
                }}
              />
            </div>
            <div className="flex items-center gap-2">
              {t('LoRA Model')}
              <Select
                style={{ minWidth: 0 }}
                className="grow"
                value={loraParams.loraLoad}
                onChange={(e, data) => {
                  setLoraParams({
                    loraLoad: data.value,
                  })
                }}
              >
                <option value="">{t('None')}</option>
                {commonStore.loraModels.map((name, index) => (
                  <option key={index} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
              <Button
                onClick={async () => {
                  const ok = await checkDependencies(navigate)
                  if (!ok) return
                  if (loraParams.loraLoad) {
                    const outputPath = `models/${loraParams.baseModel}-LoRA-${loraParams.loraLoad}`
                    MergeLora(
                      commonStore.settings.customPythonPath,
                      !!commonStore.monitorData &&
                        commonStore.monitorData.totalVram !== 0,
                      loraParams.loraAlpha,
                      'models/' + loraParams.baseModel,
                      'lora-models/' + loraParams.loraLoad,
                      outputPath
                    )
                      .then(async () => {
                        if (!(await FileExists(outputPath))) {
                          if (
                            commonStore.platform === 'windows' ||
                            commonStore.platform === 'linux'
                          )
                            toast(
                              t('Failed to merge model') +
                                ' - ' +
                                (await GetPyError()),
                              { type: 'error' }
                            )
                        } else {
                          toast(t('Merge model successfully'), {
                            type: 'success',
                          })
                        }
                      })
                      .catch(showError)
                  } else {
                    toast(t('Please select a LoRA model'), { type: 'info' })
                  }
                }}
              >
                {t('Merge Model')}
              </Button>
            </div>
            {loraFinetuneParametersOptions.map(([key, type, name], index) => {
              return (
                <Labeled
                  key={index}
                  label={t(name)}
                  content={
                    type === 'number' ? (
                      <Input
                        type="number"
                        className="grow"
                        value={loraParams[key].toString()}
                        onChange={(e, data) => {
                          setLoraParams({
                            [key]: Number(data.value),
                          })
                        }}
                      />
                    ) : type === 'boolean' ? (
                      <Switch
                        className="grow"
                        checked={loraParams[key] as boolean}
                        onChange={(e, data) => {
                          setLoraParams({
                            [key]: data.checked,
                          })
                        }}
                      />
                    ) : type === 'string' ? (
                      <Input
                        className="grow"
                        value={loraParams[key].toString()}
                        onChange={(e, data) => {
                          setLoraParams({
                            [key]: data.value,
                          })
                        }}
                      />
                    ) : type === 'LoraFinetunePrecision' ? (
                      <Dropdown
                        style={{ minWidth: 0 }}
                        className="grow"
                        value={loraParams[key].toString()}
                        selectedOptions={[loraParams[key].toString()]}
                        onOptionSelect={(_, data) => {
                          if (data.optionText) {
                            setLoraParams({
                              precision:
                                data.optionText as LoraFinetunePrecision,
                            })
                          }
                        }}
                      >
                        <Option>bf16</Option>
                        <Option>fp16</Option>
                        <Option>tf32</Option>
                      </Dropdown>
                    ) : (
                      <div />
                    )
                  }
                />
              )
            })}
          </div>
        }
      />
      <div className="grow" />
      <div className="flex gap-2">
        <div className="grow" />
        <Button
          appearance="secondary"
          size="large"
          onClick={() => {
            WslStop()
              .then(() => {
                toast(t('Command Stopped'), { type: 'success' })
              })
              .catch(showError)
          }}
        >
          {t('Stop')}
        </Button>
        <Button appearance="primary" size="large" onClick={StartLoraFinetune}>
          {t('Train')}
        </Button>
      </div>
    </div>
  )
})

const pages: { [label: string]: TrainNavigationItem } = {
  'LoRA Finetune': {
    element: <LoraFinetune />,
  },
  WSL: {
    element: <Terminal />,
  },
}

const Train: FC = () => {
  const { t } = useTranslation()
  const [tab, setTab] = useState('LoRA Finetune')

  const selectTab: SelectTabEventHandler = (e, data) =>
    typeof data.value === 'string' ? setTab(data.value) : null

  return (
    <div className="flex h-full w-full flex-col gap-2">
      <TabList
        size="small"
        appearance="subtle"
        selectedValue={tab}
        onTabSelect={selectTab}
      >
        {Object.entries(pages).map(([label]) => (
          <Tab key={label} value={label}>
            {t(label)}
          </Tab>
        ))}
      </TabList>
      <div className="grow overflow-hidden">{pages[tab].element}</div>
    </div>
  )
}

export default Train
