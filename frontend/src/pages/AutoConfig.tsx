import { FC, useEffect, useState } from 'react'
import { Button, Checkbox, Text } from '@fluentui/react-components'
import { CheckmarkCircleFilled, CircleRegular } from '@fluentui/react-icons'
import classNames from 'classnames'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import commonStore from '../stores/commonStore'

type Functions = 'Chat' | 'Completion' | 'Composition' | 'Function Call'

type CompositionMode = 'MIDI' | 'ABC'

type HardwareType = 'GPU' | 'CPU'

const NvidiaMarks = ['Nvidia', 'NVIDIA', 'GeForce', 'RTX']

enum RunState {
  Easy = 'Your hardware can run this model easily',
  Recommand = 'Your hardware can run this model with a recommended configuration',
  Advanced = 'Your hardware can run this model with an advanced configuration',
  Hard = 'Your hardware can run this model with a hard configuration',
}

export const AutoConfig: FC = observer(() => {
  const { t } = useTranslation()
  const [selectedFunction, setSelectedFunction] = useState<Functions | null>(
    null
  )
  const [compositionMode, setCompositionMode] =
    useState<CompositionMode>('MIDI')
  const [hardwareType, setHardwareType] = useState<HardwareType | null>(null)
  const [modelIndex, setModelIndex] = useState<number | null>(null)

  const reset = () => {
    setSelectedFunction(null)
    setCompositionMode('MIDI')
    setHardwareType(null)
    setModelIndex(null)
  }

  const platform = commonStore.platform

  const {
    gpuType,
    gpuName,
    usedMemory,
    totalMemory,
    gpuUsage,
    usedVram,
    totalVram,
  } = commonStore.monitorData || {}

  const isNvidia = NvidiaMarks.includes(gpuType || '')

  const info = {
    platform,
    gpuType,
    gpuName,
    usedMemory,
    totalMemory,
    gpuUsage,
    usedVram,
    totalVram,
    isNvidia,
  }

  const debugInfo = (
    <div className={classNames('flex flex-col gap-2')}>
      <Text>{t('Debug Info')}</Text>
      <Text>Platform: {platform}</Text>
      <Text>GPU Type: {gpuType}</Text>
      <Text>GPU Name: {gpuName}</Text>
      <Text>Used Memory: {usedMemory}</Text>
      <Text>Total Memory: {totalMemory}</Text>
      <Text>GPU Usage: {gpuUsage}</Text>
      <Text>Used VRAM: {usedVram}</Text>
      <Text>Total VRAM: {totalVram}</Text>
      <Text>Is Nvidia: {isNvidia ? 'Yes' : 'No'}</Text>
    </div>
  )

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

  const [selectedIndex, setSelectedIndex] = useState(
    commonStore.currentModelConfigIndex
  )
  const [selectedConfig, setSelectedConfig] = useState(
    commonStore.modelConfigs[selectedIndex]
  )

  const sectionClassName = classNames('flex')
  const sectionContentClassName = classNames('flex gap-2')

  useEffect(() => {
    const step0Done = selectedFunction != null
    const step1Done = compositionMode != null && step0Done
    const step2Done = hardwareType != null && step1Done
    const step3Done = modelIndex != null && step2Done
    const step4Done = step3Done
    setStepDone({
      0: step0Done,
      1: step1Done,
      2: step2Done,
      3: step3Done,
      4: step4Done,
    })
  }, [selectedFunction, compositionMode, hardwareType, modelIndex])

  useEffect(() => {
    const step0Disabled = false
    const step1Disabled = selectedFunction != 'Composition'
    const step2Disabled = selectedFunction == null
    const step3Disabled = hardwareType == null
    const step4Disabled = selectedIndex == null
    setStepDisable({
      0: step0Disabled,
      1: step1Disabled,
      2: step2Disabled,
      3: step3Disabled,
      4: step4Disabled,
    })
  }, [selectedFunction, compositionMode, hardwareType, modelIndex])

  const hideComposition = selectedFunction != 'Composition'
  const finalReturn = (
    <div className={classNames('flex h-full flex-col overflow-y-auto')}>
      <Button onClick={reset}>{t('Reset')}</Button>
      <div className={sectionClassName}>
        <Indicator index={0} done={stepDone[0]} />
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
        <Indicator index={1} done={stepDone[1]} />
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
        <Indicator index={2} done={stepDone[2]} />
        <div>
          <div>{t('GPU or CPU')}</div>
          <div className={sectionContentClassName}>
            <Button
              appearance={hardwareType == 'GPU' ? 'primary' : 'secondary'}
              disabled={stepDisable[2]}
              onClick={() => setHardwareType('GPU')}
            >
              {t('GPU (Recommended)')}
            </Button>
            <Button
              appearance={hardwareType == 'CPU' ? 'primary' : 'secondary'}
              disabled={stepDisable[2]}
              onClick={() => setHardwareType('CPU')}
            >
              {t('CPU')}
            </Button>
          </div>
        </div>
      </div>
      <div className={sectionClassName}>
        <Indicator index={3} done={stepDone[3]} />
        <div>
          <div>{t('Choose the model you wants to use')}</div>
          <div
            className={sectionContentClassName + ' ' + classNames('flex-wrap')}
          >
            {Array.from({ length: 10 }).map((_, index) => {
              const modelName = 'jhwbe dwen 21 e qmhbwdjb wbelwqjd kasjd'
              const runStates = Object.values(RunState)
              const runState =
                runStates[Math.floor(Math.random() * runStates.length)]
              return (
                <Button
                  key={index}
                  disabled={stepDisable[3]}
                  className={classNames(
                    'flex h-40 w-52 flex-col items-stretch rounded-sm border bg-transparent p-0 shadow-md'
                  )}
                >
                  <div className={classNames('text-lg font-semibold')}>
                    {modelName}
                  </div>
                  <div className={classNames('h-1')} />
                  <div className={classNames('overflow-clip')}>{runState}</div>
                </Button>
              )
            })}
          </div>
        </div>
      </div>
      <div className={sectionClassName}>
        <Indicator index={4} done={stepDone[4]} last />
        <div>
          <div>{t('Have a fun')}</div>
          <div className={sectionContentClassName}>
            <Button disabled={stepDisable[4]}>{t('Start')}</Button>
            <Checkbox
              disabled={stepDisable[4]}
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
          </div>
        </div>
      </div>
    </div>
  )

  return finalReturn
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
