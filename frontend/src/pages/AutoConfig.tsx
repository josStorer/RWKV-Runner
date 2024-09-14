import { FC, useEffect, useState } from 'react'
import { Button, Checkbox } from '@fluentui/react-components'
import { CheckmarkCircleFilled, CircleRegular } from '@fluentui/react-icons'
import classNames from 'classnames'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { RunButton } from '../components/RunButton'
import { ModelConfig } from '../types/configs'
import {
  CompositionMode,
  CPUOrGPU,
  Func,
  generateStrategy,
  Resolution,
} from '../utils/generate-strategy'

export const AutoConfig: FC = observer(() => {
  const { t } = useTranslation()

  const [selectedFunction, setSelectedFunction] = useState<Func | null>(null)
  const [compositionMode, setCompositionMode] =
    useState<CompositionMode>('MIDI')
  const [cpuOrGPU, setCpuOrGPU] = useState<CPUOrGPU | null>(null)

  const [selectedResolutionIndex, setSelectedResolutionIndex] = useState<
    number | null
  >(null)

  const reset = () => {
    setSelectedFunction(null)
    setCompositionMode('MIDI')
    setCpuOrGPU(null)
    setSelectedResolutionIndex(null)
  }

  const [stepDisable, setStepDisable] = useState({
    0: false,
    1: false,
    2: false,
    3: false,
    4: false,
  })

  const [stepDone, setStepDone] = useState({
    0: false,
    1: false,
    2: false,
    3: false,
    4: false,
  })

  const sectionClassName = classNames('flex')
  const sectionContentClassName = classNames('flex gap-2')

  useEffect(() => {
    const step0Done = selectedFunction != null
    const step1Done = compositionMode != null && step0Done
    const step2Done = cpuOrGPU != null && step1Done
    const step3Done = selectedResolutionIndex != null && step2Done
    const step4Done = step3Done
    setStepDone({
      0: step0Done,
      1: step1Done,
      2: step2Done,
      3: step3Done,
      4: step4Done,
    })
  }, [selectedFunction, compositionMode, cpuOrGPU, selectedResolutionIndex])

  useEffect(() => {
    setSelectedResolutionIndex(null)
  }, [selectedFunction, compositionMode, cpuOrGPU])

  useEffect(() => {
    const step0Disabled = false
    const step1Disabled = selectedFunction != 'Composition'
    const step2Disabled = selectedFunction == null
    const step3Disabled = cpuOrGPU == null
    const step4Disabled = selectedResolutionIndex == null
    setStepDisable({
      0: step0Disabled,
      1: step1Disabled,
      2: step2Disabled,
      3: step3Disabled,
      4: step4Disabled,
    })
  }, [selectedFunction, compositionMode, cpuOrGPU, selectedResolutionIndex])

  const hideComposition = selectedFunction !== 'Composition'

  const resolutions: Resolution[] = generateStrategy(
    selectedFunction,
    compositionMode,
    cpuOrGPU
  )

  useEffect(() => {
    const gpuAvailable = selectedFunction !== 'Composition'
    if (!gpuAvailable && cpuOrGPU === 'GPU') {
      setCpuOrGPU('CPU')
    }
  }, [selectedFunction])

  const onStartButtonClick = () => {
    console.log('onStartButtonClick')
  }

  const [config, setConfig] = useState<ModelConfig | null>(null)

  useEffect(() => {
    if (
      selectedResolutionIndex !== null &&
      selectedResolutionIndex < resolutions.length
    ) {
      const resolution = resolutions[selectedResolutionIndex]
      const modelConfig = resolution.modelConfig
      if (modelConfig) {
        setConfig(modelConfig)
      }
    }
  }, [selectedResolutionIndex, resolutions])

  return (
    <div className={classNames('flex h-full flex-col overflow-y-auto')}>
      <Button onClick={reset}>{t('Reset')}</Button>
      <div className={sectionClassName}>
        <Indicator
          index={0}
          done={stepDone[0]}
        />
        <div>
          <div>{t('Choose the function you wants to invoke')}</div>
          <div className={sectionContentClassName}>
            <Button
              appearance={selectedFunction == 'Chat' ? 'primary' : 'secondary'}
              onClick={() => setSelectedFunction('Chat')}
            >
              {t('Chat')}
            </Button>
            <Button
              appearance={
                selectedFunction == 'Completion' ? 'primary' : 'secondary'
              }
              onClick={() => setSelectedFunction('Completion')}
            >
              {t('Completion')}
            </Button>
            <Button
              appearance={
                selectedFunction == 'Composition' ? 'primary' : 'secondary'
              }
              onClick={() => setSelectedFunction('Composition')}
            >
              {t('Composition')}
            </Button>
            <Button
              appearance={
                selectedFunction == 'Function Call' ? 'primary' : 'secondary'
              }
              onClick={() => setSelectedFunction('Function Call')}
            >
              {t('Function Call')}
            </Button>
          </div>
        </div>
      </div>
      <div
        className={
          sectionClassName + ' ' + classNames({ hidden: hideComposition })
        }
      >
        <Indicator
          index={1}
          done={stepDone[1]}
        />
        <div>
          <div>{t('Composition Mode')}</div>
          <div className={sectionContentClassName}>
            <Button
              appearance={compositionMode == 'MIDI' ? 'primary' : 'secondary'}
              disabled={stepDisable[1]}
              onClick={() => setCompositionMode('MIDI')}
            >
              {t('MIDI')}
            </Button>
            <Button
              appearance={compositionMode == 'ABC' ? 'primary' : 'secondary'}
              disabled={stepDisable[1]}
              onClick={() => setCompositionMode('ABC')}
            >
              {t('ABC')}
            </Button>
          </div>
        </div>
      </div>
      <div className={sectionClassName}>
        <Indicator
          index={2}
          done={stepDone[2]}
        />
        <div>
          <div>{t('GPU or CPU')}</div>
          <div className={sectionContentClassName}>
            <Button
              appearance={cpuOrGPU == 'GPU' ? 'primary' : 'secondary'}
              disabled={stepDisable[2] || selectedFunction === 'Composition'}
              onClick={() => setCpuOrGPU('GPU')}
            >
              {t('GPU (Recommended)')}
            </Button>
            <Button
              appearance={cpuOrGPU == 'CPU' ? 'primary' : 'secondary'}
              disabled={stepDisable[2]}
              onClick={() => setCpuOrGPU('CPU')}
            >
              {t('CPU')}
            </Button>
          </div>
        </div>
      </div>
      <div className={sectionClassName}>
        <Indicator
          index={3}
          done={stepDone[3]}
        />
        <div>
          <div>{t('Choose the model and strategy you wants to use')}</div>
          <div
            className={sectionContentClassName + ' ' + classNames('flex-wrap')}
          >
            {resolutions.map((item, index) => {
              const {
                comments,
                modelName,
                calculateByPrecision,
                requirements,
                usingGPU,
                recommendLevel,
                modelConfig,
              } = item

              const requirement = requirements[calculateByPrecision]

              return (
                <Button
                  appearance={
                    selectedResolutionIndex === index ? 'primary' : 'secondary'
                  }
                  key={index}
                  disabled={stepDisable[3] || recommendLevel <= -2}
                  className={classNames(
                    'flex w-52 flex-col items-stretch rounded-sm p-0 text-left shadow-md'
                  )}
                  onClick={() => setSelectedResolutionIndex(index)}
                >
                  <div className={classNames('text-lg font-semibold')}>
                    {modelName}
                  </div>
                  <div className={classNames('h-1')} />
                  <div
                    className={classNames('overflow-clip text-left text-xs')}
                  >
                    {comments}
                  </div>
                  <div className={classNames('h-1')} />
                  <div className={classNames('text-xs')}>
                    {usingGPU ? 'GPU' : 'CPU'}
                    <div />
                    {calculateByPrecision}
                    <div />
                    {usingGPU
                      ? t('Estimated VRAM usage: ')
                      : t('Estimated RAM usage: ')}
                    {requirement?.toFixed(1) + 'GB'}
                    <div />
                    {modelConfig?.modelParameters?.device}
                  </div>
                </Button>
              )
            })}
          </div>
        </div>
      </div>
      <div className={sectionClassName}>
        <Indicator
          index={4}
          done={stepDone[4]}
          last
        />
        <div>
          <div>{t('Have a fun')}</div>
          <div className={sectionContentClassName}>
            <RunButton
              onClickRun={onStartButtonClick}
              disabled={stepDisable[4] || !config}
              config={config}
            />
            <Checkbox
              disabled={stepDisable[4] || !config}
              className="select-none"
              size="large"
              label={t('Enable WebUI')}
              checked={config?.enableWebUI}
              onChange={(_, data) => {
                setConfig({
                  ...config!,
                  enableWebUI: data.checked as boolean,
                })
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
})

const Indicator: FC<{ index: number; done: boolean; last?: boolean }> = ({
  index,
  done,
  last = false,
}) => {
  const icon = done ? (
    <CheckmarkCircleFilled className={classNames('text-3xl text-green-500')} />
  ) : (
    <CircleRegular className={classNames('text-3xl')} />
  )
  const first = index == 0
  const classNameTop = classNames('h-1 w-0.5', {
    'bg-green-500': done,
    'bg-gray-500': !done,
    'bg-transparent': first,
  })
  const classNameBottom = classNames('h-full w-0.5', {
    'bg-green-500': done && !last,
    'bg-gray-500': !done && !last,
    'bg-transparent': last,
  })
  const topLine = <div className={classNameTop} />
  const bottomLine = <div className={classNameBottom} />
  return (
    <div className={classNames('flex flex-col items-center')}>
      {topLine}
      <div>{icon}</div>
      {bottomLine}
    </div>
  )
}
