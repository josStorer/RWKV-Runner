import React, { FC, useCallback, useEffect, useRef } from 'react'
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Checkbox,
  Dropdown,
  Input,
  Label,
  Link,
  Option,
  PresenceBadge,
  Select,
  Switch,
  Text,
  Tooltip,
} from '@fluentui/react-components'
import {
  AddCircle20Regular,
  DataUsageSettings20Regular,
  Delete20Regular,
  Save20Regular,
} from '@fluentui/react-icons'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router'
import { toast } from 'react-toastify'
import { useMediaQuery } from 'usehooks-ts'
import { BrowserOpenURL } from '../../wailsjs/runtime'
import { updateConfig } from '../apis'
import strategyZhImg from '../assets/images/strategy_zh.jpg'
import strategyImg from '../assets/images/strategy.jpg'
import { Labeled } from '../components/Labeled'
import { NumberInput } from '../components/NumberInput'
import { Page } from '../components/Page'
import { ResetConfigsButton } from '../components/ResetConfigsButton'
import { RunButton } from '../components/RunButton'
import { Section } from '../components/Section'
import { ToolTipButton } from '../components/ToolTipButton'
import { ValuedSlider } from '../components/ValuedSlider'
import commonStore from '../stores/commonStore'
import {
  ApiParameters,
  Device,
  ModelParameters,
  Precision,
} from '../types/configs'
import { getStrategy, isDynamicStateSupported } from '../utils'
import {
  convertModel,
  convertToGGML,
  convertToSt,
} from '../utils/convert-model'
import { defaultPenaltyDecay } from './defaultConfigs'

const ConfigSelector: FC<{
  selectedIndex: number
  updateSelectedIndex: (i: number) => void
}> = observer(({ selectedIndex, updateSelectedIndex }) => {
  return (
    <Dropdown
      style={{ minWidth: 0 }}
      className="grow"
      value={commonStore.modelConfigs[selectedIndex].name}
      selectedOptions={[selectedIndex.toString()]}
      onOptionSelect={(_, data) => {
        if (data.optionValue) {
          updateSelectedIndex(Number(data.optionValue))
        }
      }}
    >
      {commonStore.modelConfigs.map((config, index) => (
        <Option key={index} value={index.toString()} text={config.name}>
          <div className="flex grow justify-between">
            {config.name}
            {commonStore.modelSourceList.find(
              (item) => item.name === config.modelParameters.modelName
            )?.isComplete && <PresenceBadge status="available" />}
          </div>
        </Option>
      ))}
    </Dropdown>
  )
})

const Configs: FC = observer(() => {
  const { t } = useTranslation()
  const [selectedIndex, setSelectedIndex] = React.useState(
    commonStore.currentModelConfigIndex
  )
  const [selectedConfig, setSelectedConfig] = React.useState(
    commonStore.modelConfigs[selectedIndex]
  )
  const [displayStrategyImg, setDisplayStrategyImg] = React.useState(false)
  const advancedHeaderRef1 = useRef<HTMLDivElement>(null)
  const advancedHeaderRef2 = useRef<HTMLDivElement>(null)
  const mq = useMediaQuery('(min-width: 640px)')
  const navigate = useNavigate()
  const port = selectedConfig.apiParameters.apiPort

  useEffect(() => {
    if (advancedHeaderRef1.current)
      (
        advancedHeaderRef1.current.firstElementChild as HTMLElement
      ).style.padding = '0'
    if (advancedHeaderRef2.current)
      (
        advancedHeaderRef2.current.firstElementChild as HTMLElement
      ).style.padding = '0'
  }, [])

  const updateSelectedIndex = useCallback((newIndex: number) => {
    setSelectedIndex(newIndex)
    setSelectedConfig(commonStore.modelConfigs[newIndex])

    // if you don't want to update the config used by the current startup in real time, comment out this line
    commonStore.setCurrentConfigIndex(newIndex)
  }, [])

  const setSelectedConfigName = (newName: string) => {
    setSelectedConfig({ ...selectedConfig, name: newName })
  }

  const setSelectedConfigApiParams = (newParams: Partial<ApiParameters>) => {
    setSelectedConfig({
      ...selectedConfig,
      apiParameters: {
        ...selectedConfig.apiParameters,
        ...newParams,
      },
    })
  }

  const setSelectedConfigModelParams = (
    newParams: Partial<ModelParameters>
  ) => {
    setSelectedConfig({
      ...selectedConfig,
      modelParameters: {
        ...selectedConfig.modelParameters,
        ...newParams,
      },
    })
  }

  const onClickSave = () => {
    commonStore.setModelConfig(selectedIndex, selectedConfig)
    const webgpu = selectedConfig.modelParameters.device === 'WebGPU'
    if (!webgpu) {
      // When clicking RunButton in Configs page, updateConfig will be called twice,
      // because there are also RunButton in other pages, and the calls to updateConfig in both places are necessary.
      updateConfig(t, {
        max_tokens: selectedConfig.apiParameters.maxResponseToken,
        temperature: selectedConfig.apiParameters.temperature,
        top_p: selectedConfig.apiParameters.topP,
        presence_penalty: selectedConfig.apiParameters.presencePenalty,
        frequency_penalty: selectedConfig.apiParameters.frequencyPenalty,
        penalty_decay: selectedConfig.apiParameters.penaltyDecay,
        global_penalty: selectedConfig.apiParameters.globalPenalty,
        state: selectedConfig.apiParameters.stateModel,
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
            toast(t('File format of the model or state model not supported'), {
              type: 'error',
            })
          else toast(error, { type: 'error' })
        }
      })
    }
    toast(t('Config Saved'), { autoClose: 300, type: 'success' })
  }

  return (
    <Page
      title={t('Configs')}
      content={
        <div className="flex flex-col gap-2 overflow-hidden">
          <div className="flex items-center gap-2">
            <ConfigSelector
              selectedIndex={selectedIndex}
              updateSelectedIndex={updateSelectedIndex}
            />
            <ToolTipButton
              desc={t('New Config')}
              icon={<AddCircle20Regular />}
              onClick={() => {
                commonStore.createModelConfig()
                updateSelectedIndex(commonStore.modelConfigs.length - 1)
              }}
            />
            <ToolTipButton
              desc={t('Delete Config')}
              icon={<Delete20Regular />}
              onClick={() => {
                commonStore.deleteModelConfig(selectedIndex)
                updateSelectedIndex(
                  Math.min(selectedIndex, commonStore.modelConfigs.length - 1)
                )
              }}
            />
            <ResetConfigsButton
              afterConfirm={() => {
                setSelectedIndex(0)
                setSelectedConfig(commonStore.modelConfigs[0])
              }}
            />
            <ToolTipButton
              desc={mq ? '' : t('Save Config')}
              icon={<Save20Regular />}
              text={mq ? t('Save Config') : null}
              onClick={onClickSave}
            />
          </div>
          <div className="flex items-center gap-4">
            <Label>{t('Config Name')}</Label>
            <Input
              className="grow"
              value={selectedConfig.name}
              onChange={(e, data) => {
                setSelectedConfigName(data.value)
              }}
            />
          </div>
          <div className="flex flex-col gap-2 overflow-y-hidden">
            <Section
              title={t('Default API Parameters')}
              desc={t(
                'Hover your mouse over the text to view a detailed description. Settings marked with * will take effect immediately after being saved.'
              )}
              content={
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <Labeled
                    label={t('API Port')}
                    desc={
                      t(
                        'Open the following URL with your browser to view the API documentation'
                      ) +
                      `: http://127.0.0.1:${port}/docs. ` +
                      t(
                        "This tool's API is compatible with OpenAI API. It can be used with any ChatGPT tool you like. Go to the settings of some ChatGPT tool, replace the 'https://api.openai.com' part in the API address with '"
                      ) +
                      `http://127.0.0.1:${port}` +
                      "'."
                    }
                    content={
                      <NumberInput
                        value={port}
                        min={1}
                        max={65535}
                        step={1}
                        onChange={(e, data) => {
                          setSelectedConfigApiParams({
                            apiPort: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    label={t('Max Response Token') + ' *'}
                    desc={t(
                      'By default, the maximum number of tokens that can be answered in a single response, it can be changed by the user by specifying API parameters.'
                    )}
                    content={
                      <ValuedSlider
                        value={selectedConfig.apiParameters.maxResponseToken}
                        min={100}
                        max={8100}
                        step={100}
                        input
                        onChange={(e, data) => {
                          setSelectedConfigApiParams({
                            maxResponseToken: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    label={t('Temperature') + ' *'}
                    desc={t(
                      "Sampling temperature, it's like giving alcohol to a model, the higher the stronger the randomness and creativity, while the lower, the more focused and deterministic it will be."
                    )}
                    content={
                      <ValuedSlider
                        value={selectedConfig.apiParameters.temperature}
                        min={0}
                        max={3}
                        step={0.1}
                        input
                        onChange={(e, data) => {
                          setSelectedConfigApiParams({
                            temperature: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    label={t('Top_P') + ' *'}
                    desc={t(
                      'Just like feeding sedatives to the model. Consider the results of the top n% probability mass, 0.1 considers the top 10%, with higher quality but more conservative, 1 considers all results, with lower quality but more diverse.'
                    )}
                    content={
                      <ValuedSlider
                        value={selectedConfig.apiParameters.topP}
                        min={0}
                        max={1}
                        step={0.05}
                        input
                        onChange={(e, data) => {
                          setSelectedConfigApiParams({
                            topP: data.value,
                          })
                        }}
                      />
                    }
                  />
                  {isDynamicStateSupported(selectedConfig) && (
                    <div className="flex min-w-0 items-center gap-2 sm:col-span-2">
                      <Tooltip
                        content={
                          <div>
                            {t('State-tuned Model')}, {t('See More')}:{' '}
                            <Link
                              onClick={() =>
                                BrowserOpenURL(
                                  'https://github.com/BlinkDL/RWKV-LM#state-tuning-tuning-the-initial-state-zero-inference-overhead'
                                )
                              }
                            >
                              {
                                'https://github.com/BlinkDL/RWKV-LM#state-tuning-tuning-the-initial-state-zero-inference-overhead'
                              }
                            </Link>
                          </div>
                        }
                        showDelay={0}
                        hideDelay={0}
                        relationship="description"
                      >
                        <div className="shrink-0">
                          {t('State Model') + ' *'}
                        </div>
                      </Tooltip>
                      <Select
                        style={{ minWidth: 0 }}
                        className="grow"
                        value={selectedConfig.apiParameters.stateModel}
                        onChange={(e, data) => {
                          setSelectedConfigApiParams({
                            stateModel: data.value,
                          })
                        }}
                      >
                        <option key={-1} value={''}>
                          {t('None')}
                        </option>
                        {commonStore.stateModels.map((modelName, index) => (
                          <option key={index} value={modelName}>
                            {modelName}
                          </option>
                        ))}
                      </Select>
                    </div>
                  )}
                  <Accordion
                    className="sm:col-span-2"
                    collapsible
                    openItems={!commonStore.apiParamsCollapsed && 'advanced'}
                    onToggle={(e, data) => {
                      if (data.value === 'advanced')
                        commonStore.setApiParamsCollapsed(
                          !commonStore.apiParamsCollapsed
                        )
                    }}
                  >
                    <AccordionItem value="advanced">
                      <AccordionHeader ref={advancedHeaderRef1} size="small">
                        {t('Advanced')}
                      </AccordionHeader>
                      <AccordionPanel>
                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                          <Labeled
                            label={t('Presence Penalty') + ' *'}
                            desc={t(
                              "Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."
                            )}
                            content={
                              <ValuedSlider
                                value={
                                  selectedConfig.apiParameters.presencePenalty
                                }
                                min={-2}
                                max={2}
                                step={0.1}
                                input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    presencePenalty: data.value,
                                  })
                                }}
                              />
                            }
                          />
                          <Labeled
                            label={t('Frequency Penalty') + ' *'}
                            desc={t(
                              "Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."
                            )}
                            content={
                              <ValuedSlider
                                value={
                                  selectedConfig.apiParameters.frequencyPenalty
                                }
                                min={-2}
                                max={2}
                                step={0.1}
                                input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    frequencyPenalty: data.value,
                                  })
                                }}
                              />
                            }
                          />
                          <Labeled
                            label={
                              t('Penalty Decay') +
                              (!selectedConfig.apiParameters.penaltyDecay ||
                              selectedConfig.apiParameters.penaltyDecay ===
                                defaultPenaltyDecay
                                ? ` (${t('Default')})`
                                : '') +
                              ' *'
                            }
                            desc={t(
                              "If you don't know what it is, keep it default."
                            )}
                            content={
                              <ValuedSlider
                                value={
                                  selectedConfig.apiParameters.penaltyDecay ||
                                  defaultPenaltyDecay
                                }
                                min={0.99}
                                max={0.999}
                                step={0.001}
                                toFixed={3}
                                input
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    penaltyDecay: data.value,
                                  })
                                }}
                              />
                            }
                          />
                          <Labeled
                            label={t('Global Penalty') + ' *'}
                            desc={t(
                              'When generating a response, whether to include the submitted prompt as a penalty factor. By turning this off, you will get the same generated results as official RWKV Gradio. If you find duplicate results in the generated results, turning this on can help avoid generating duplicates.'
                            )}
                            content={
                              <Switch
                                checked={
                                  selectedConfig.apiParameters.globalPenalty
                                }
                                onChange={(e, data) => {
                                  setSelectedConfigApiParams({
                                    globalPenalty: data.checked,
                                  })
                                }}
                              />
                            }
                          />
                        </div>
                      </AccordionPanel>
                    </AccordionItem>
                  </Accordion>
                </div>
              }
            />
            <Section
              title={t('Model Parameters')}
              content={
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <div className="sm:col-span-2">
                    <div className="flex flex-col gap-2 sm:flex-row">
                      <div className="flex min-w-0 grow items-center gap-2">
                        <div className="shrink-0">{t('Model')}</div>
                        <Select
                          style={{ minWidth: 0 }}
                          className="grow"
                          value={selectedConfig.modelParameters.modelName}
                          onChange={(e, data) => {
                            const modelSource =
                              commonStore.modelSourceList.find(
                                (item) => item.name === data.value
                              )
                            if (modelSource?.customTokenizer)
                              setSelectedConfigModelParams({
                                modelName: data.value,
                                useCustomTokenizer: true,
                                customTokenizer: modelSource?.customTokenizer,
                              })
                            // prevent customTokenizer from being overwritten
                            else
                              setSelectedConfigModelParams({
                                modelName: data.value,
                                useCustomTokenizer: false,
                              })
                          }}
                        >
                          {!commonStore.modelSourceList.find(
                            (item) =>
                              item.name ===
                              selectedConfig.modelParameters.modelName
                          )?.isComplete && (
                            <option
                              key={-1}
                              value={selectedConfig.modelParameters.modelName}
                            >
                              {selectedConfig.modelParameters.modelName}
                            </option>
                          )}
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
                      {!selectedConfig.modelParameters.device.startsWith(
                        'WebGPU'
                      ) ? (
                        selectedConfig.modelParameters.device !==
                        'CPU (rwkv.cpp)' ? (
                          <ToolTipButton
                            text={t('Convert')}
                            className="shrink-0"
                            desc={t(
                              'Convert model with these configs. Using a converted model will greatly improve the loading speed, but model parameters of the converted model cannot be modified.'
                            )}
                            onClick={() =>
                              convertModel(selectedConfig, navigate)
                            }
                          />
                        ) : (
                          <ToolTipButton
                            text={t('Convert To GGML Format')}
                            className="shrink-0"
                            desc=""
                            onClick={() =>
                              convertToGGML(selectedConfig, navigate)
                            }
                          />
                        )
                      ) : (
                        <ToolTipButton
                          text={t('Convert To Safe Tensors Format')}
                          className="shrink-0"
                          desc=""
                          onClick={() => convertToSt(selectedConfig, navigate)}
                        />
                      )}
                    </div>
                  </div>
                  <Labeled
                    label={t('Strategy')}
                    content={
                      <Dropdown
                        style={{ minWidth: 0 }}
                        className="grow"
                        value={t(selectedConfig.modelParameters.device)!}
                        selectedOptions={[
                          selectedConfig.modelParameters.device,
                        ]}
                        onOptionSelect={(_, data) => {
                          if (data.optionValue) {
                            setSelectedConfigModelParams({
                              device: data.optionValue as Device,
                            })
                          }
                        }}
                      >
                        <Option value="CPU">CPU</Option>
                        <Option value="CPU (rwkv.cpp)">
                          {t('CPU (rwkv.cpp, Faster)')!}
                        </Option>
                        {/*{commonStore.platform === 'darwin' && <Option value="MPS">MPS</Option>}*/}
                        <Option value="CUDA">CUDA</Option>
                        {/*<Option value="CUDA-Beta">{t('CUDA (Beta, Faster)')!}</Option>*/}
                        <Option value="WebGPU">WebGPU</Option>
                        <Option value="WebGPU (Python)">WebGPU (Python)</Option>
                        <Option value="Custom">{t('Custom')!}</Option>
                      </Dropdown>
                    }
                  />
                  {selectedConfig.modelParameters.device !== 'Custom' && (
                    <Labeled
                      label={t('Precision')}
                      desc={t(
                        'int8 uses less VRAM, but has slightly lower quality. fp16 has higher quality.'
                      )}
                      content={
                        <Dropdown
                          style={{ minWidth: 0 }}
                          className="grow"
                          value={selectedConfig.modelParameters.precision}
                          selectedOptions={[
                            selectedConfig.modelParameters.precision,
                          ]}
                          onOptionSelect={(_, data) => {
                            if (data.optionText) {
                              setSelectedConfigModelParams({
                                precision: data.optionText as Precision,
                              })
                            }
                          }}
                        >
                          {selectedConfig.modelParameters.device !== 'CPU' &&
                            selectedConfig.modelParameters.device !== 'MPS' && (
                              <Option>fp16</Option>
                            )}
                          {selectedConfig.modelParameters.device !==
                            'CPU (rwkv.cpp)' && <Option>int8</Option>}
                          {selectedConfig.modelParameters.device.startsWith(
                            'WebGPU'
                          ) && <Option>nf4</Option>}
                          {selectedConfig.modelParameters.device !==
                            'CPU (rwkv.cpp)' &&
                            !selectedConfig.modelParameters.device.startsWith(
                              'WebGPU'
                            ) && <Option>fp32</Option>}
                          {selectedConfig.modelParameters.device ===
                            'CPU (rwkv.cpp)' && <Option>Q5_1</Option>}
                        </Dropdown>
                      }
                    />
                  )}
                  {selectedConfig.modelParameters.device.startsWith('CUDA') && (
                    <Labeled
                      label={t('Current Strategy')}
                      content={<Text> {getStrategy(selectedConfig)} </Text>}
                    />
                  )}
                  {selectedConfig.modelParameters.device.startsWith('CUDA') && (
                    <Labeled
                      label={t('Stored Layers')}
                      desc={t(
                        'Number of the neural network layers loaded into VRAM, the more you load, the faster the speed, but it consumes more VRAM. (If your VRAM is not enough, it will fail to load)'
                      )}
                      content={
                        <ValuedSlider
                          value={selectedConfig.modelParameters.storedLayers}
                          min={0}
                          max={selectedConfig.modelParameters.maxStoredLayers}
                          step={1}
                          input
                          onChange={(e, data) => {
                            setSelectedConfigModelParams({
                              storedLayers: data.value,
                            })
                          }}
                        />
                      }
                    />
                  )}
                  {selectedConfig.modelParameters.device.startsWith(
                    'WebGPU'
                  ) && (
                    <Labeled
                      label={t('Parallel Token Chunk Size')}
                      desc={t(
                        'Maximum tokens to be processed in parallel at once. For high end GPUs, this could be 64 or 128 (faster).'
                      )}
                      content={
                        <ValuedSlider
                          value={
                            selectedConfig.modelParameters.tokenChunkSize || 32
                          }
                          min={16}
                          max={256}
                          step={16}
                          input
                          onChange={(e, data) => {
                            setSelectedConfigModelParams({
                              tokenChunkSize: data.value,
                            })
                          }}
                        />
                      }
                    />
                  )}
                  {selectedConfig.modelParameters.device.startsWith(
                    'WebGPU'
                  ) && (
                    <Labeled
                      label={t('Quantized Layers')}
                      desc={t(
                        'Number of the neural network layers quantized with current precision, the more you quantize, the lower the VRAM usage, but the quality correspondingly decreases.'
                      )}
                      content={
                        <ValuedSlider
                          disabled={
                            selectedConfig.modelParameters.precision !==
                              'int8' &&
                            selectedConfig.modelParameters.precision !== 'nf4'
                          }
                          value={
                            selectedConfig.modelParameters.precision === 'int8'
                              ? selectedConfig.modelParameters
                                  .quantizedLayers || 31
                              : selectedConfig.modelParameters.precision ===
                                  'nf4'
                                ? selectedConfig.modelParameters
                                    .quantizedLayers || 26
                                : selectedConfig.modelParameters.maxStoredLayers
                          }
                          min={0}
                          max={selectedConfig.modelParameters.maxStoredLayers}
                          step={1}
                          input
                          onChange={(e, data) => {
                            setSelectedConfigModelParams({
                              quantizedLayers: data.value,
                            })
                          }}
                        />
                      }
                    />
                  )}
                  {selectedConfig.modelParameters.device.startsWith('CUDA') && (
                    <div />
                  )}
                  {displayStrategyImg && (
                    <img
                      style={{ width: '80vh', height: 'auto', zIndex: 100 }}
                      className="fixed left-0 top-0 select-none rounded-xl"
                      src={
                        commonStore.settings.language === 'zh'
                          ? strategyZhImg
                          : strategyImg
                      }
                    />
                  )}
                  {selectedConfig.modelParameters.device === 'Custom' && (
                    <Labeled
                      label="Strategy"
                      onMouseEnter={() => setDisplayStrategyImg(true)}
                      onMouseLeave={() => setDisplayStrategyImg(false)}
                      content={
                        <Input
                          className="grow"
                          placeholder={
                            commonStore.platform !== 'darwin'
                              ? 'cuda:0 fp16 *20 -> cuda:1 fp16'
                              : 'mps fp32'
                          }
                          value={selectedConfig.modelParameters.customStrategy}
                          onChange={(e, data) => {
                            setSelectedConfigModelParams({
                              customStrategy: data.value,
                            })
                          }}
                        />
                      }
                    />
                  )}
                  {selectedConfig.modelParameters.device === 'Custom' && (
                    <div />
                  )}
                  {(selectedConfig.modelParameters.device.startsWith('CUDA') ||
                    selectedConfig.modelParameters.device === 'Custom') && (
                    <Labeled
                      label={t('Use Custom CUDA kernel to Accelerate')}
                      desc={t(
                        'Enabling this option can greatly improve inference speed and save some VRAM, but there may be compatibility issues (output garbled). If it fails to start, please turn off this option, or try to upgrade your gpu driver.'
                      )}
                      content={
                        <Switch
                          checked={selectedConfig.modelParameters.useCustomCuda}
                          onChange={(e, data) => {
                            setSelectedConfigModelParams({
                              useCustomCuda: data.checked,
                            })
                          }}
                        />
                      }
                    />
                  )}
                  {selectedConfig.modelParameters.device !== 'WebGPU' && (
                    <Accordion
                      className="sm:col-span-2"
                      collapsible
                      openItems={
                        !commonStore.modelParamsCollapsed && 'advanced'
                      }
                      onToggle={(e, data) => {
                        if (data.value === 'advanced')
                          commonStore.setModelParamsCollapsed(
                            !commonStore.modelParamsCollapsed
                          )
                      }}
                    >
                      <AccordionItem value="advanced">
                        <AccordionHeader ref={advancedHeaderRef2} size="small">
                          {t('Advanced')}
                        </AccordionHeader>
                        <AccordionPanel>
                          <div className="flex flex-col">
                            <div className="flex grow">
                              <Checkbox
                                className="select-none"
                                size="large"
                                label={t('Use Custom Tokenizer')}
                                checked={
                                  selectedConfig.modelParameters
                                    .useCustomTokenizer
                                }
                                onChange={(_, data) => {
                                  setSelectedConfigModelParams({
                                    useCustomTokenizer: data.checked as boolean,
                                  })
                                }}
                              />
                              <Input
                                className="grow"
                                placeholder={
                                  t(
                                    'Tokenizer Path (e.g. backend-python/rwkv_pip/20B_tokenizer.json or rwkv_vocab_v20230424.txt)'
                                  )!
                                }
                                value={
                                  selectedConfig.modelParameters.customTokenizer
                                }
                                onChange={(e, data) => {
                                  setSelectedConfigModelParams({
                                    customTokenizer: data.value,
                                  })
                                }}
                              />
                            </div>
                          </div>
                        </AccordionPanel>
                      </AccordionItem>
                    </Accordion>
                  )}
                </div>
              }
            />
            {mq && <div style={{ minHeight: '30px' }} />}
          </div>
          <div className="bottom-2 right-2 flex flex-row-reverse sm:fixed">
            <div className="flex gap-2">
              {selectedConfig.modelParameters.device !== 'WebGPU' && (
                <Checkbox
                  className="select-none"
                  size="large"
                  label={t('Enable WebUI')}
                  checked={selectedConfig.enableWebUI}
                  onChange={(_, data) => {
                    setSelectedConfig({
                      ...selectedConfig,
                      enableWebUI: data.checked as boolean,
                    })
                  }}
                />
              )}
              <RunButton onClickRun={onClickSave} />
            </div>
          </div>
        </div>
      }
    />
  )
})

export default Configs
