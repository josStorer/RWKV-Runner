import React, { FC, useEffect, useRef, useState } from 'react'
import {
  Button,
  Card,
  DialogTrigger,
  Slider,
  Text,
  Tooltip,
} from '@fluentui/react-components'
import {
  Add16Regular,
  ArrowAutofitWidth20Regular,
  ArrowUpload16Regular,
  Delete16Regular,
  MusicNote220Regular,
  Pause16Regular,
  Play16Filled,
  Play16Regular,
  Record16Regular,
  Stop16Filled,
} from '@fluentui/react-icons'
import classnames from 'classnames'
import { observer } from 'mobx-react-lite'
import Draggable from 'react-draggable'
import { useTranslation } from 'react-i18next'
import { toast } from 'react-toastify'
import { useWindowSize } from 'usehooks-ts'
import { v4 as uuid } from 'uuid'
import { PlayNote } from '../../../wailsjs/go/backend_golang/App'
import { ToolTipButton } from '../../components/ToolTipButton'
import commonStore, { ModelStatus } from '../../stores/commonStore'
import {
  InstrumentType,
  InstrumentTypeNameMap,
  InstrumentTypeTokenMap,
  MidiMessage,
  tracksMinimalTotalTime,
} from '../../types/composition'
import {
  flushMidiRecordingContent,
  getMidiRawContentMainInstrument,
  getMidiRawContentTime,
  getReqUrl,
  OpenFileDialog,
  refreshTracksTotalTime,
} from '../../utils'

const snapValue = 25
const minimalMoveTime = 8 // 1000/125=8ms wait_events=125
const scaleMin = 0.05
const scaleMax = 3
const baseMoveTime = Math.round(minimalMoveTime / scaleMin)

const velocityEvents = 128
const velocityBins = 12
const velocityExp = 0.5

const minimalTrackWidth = 80
const trackInitOffsetPx = 10
const pixelFix = 0.5
const topToArrowIcon = 19
const arrowIconToTracks = 23

const velocityToBin = (velocity: number) => {
  velocity = Math.max(0, Math.min(velocity, velocityEvents - 1))
  const binsize = velocityEvents / (velocityBins - 1)
  return Math.ceil(
    (velocityEvents *
      ((Math.pow(velocityExp, velocity / velocityEvents) - 1.0) /
        (velocityExp - 1.0))) /
      binsize
  )
}

const binToVelocity = (bin: number) => {
  const binsize = velocityEvents / (velocityBins - 1)
  return Math.max(
    0,
    Math.ceil(
      velocityEvents *
        (Math.log(((velocityExp - 1) * binsize * bin) / velocityEvents + 1) /
          Math.log(velocityExp)) -
        1
    )
  )
}

const tokenToMidiMessage = (token: string): MidiMessage | null => {
  if (token.startsWith('<')) return null
  if (token.startsWith('t') && !token.startsWith('t:')) {
    return {
      messageType: 'ElapsedTime',
      value: parseInt(token.substring(1)) * minimalMoveTime,
      channel: 0,
      note: 0,
      velocity: 0,
      control: 0,
      instrument: 0,
    }
  }
  const instrument: InstrumentType = InstrumentTypeTokenMap.findIndex((t) =>
    token.startsWith(t + ':')
  )
  if (instrument >= 0) {
    const parts = token.split(':')
    if (parts.length !== 3) return null
    const note = parseInt(parts[1], 16)
    const velocity = parseInt(parts[2], 16)
    if (velocity < 0 || velocity > 127) return null
    if (velocity === 0)
      return {
        messageType: 'NoteOff',
        note: note,
        instrument: instrument,
        channel: 0,
        velocity: 0,
        control: 0,
        value: 0,
      }
    return {
      messageType: 'NoteOn',
      note: note,
      velocity: binToVelocity(velocity),
      instrument: instrument,
      channel: 0,
      control: 0,
      value: 0,
    } as MidiMessage
  }
  return null
}

const midiMessageToToken = (msg: MidiMessage) => {
  if (msg.messageType === 'NoteOn' || msg.messageType === 'NoteOff') {
    const instrument = InstrumentTypeTokenMap[msg.instrument]
    const note = msg.note.toString(16)
    const velocity = velocityToBin(msg.velocity).toString(16)
    return `${instrument}:${note}:${velocity} `
  } else if (msg.messageType === 'ElapsedTime') {
    let time = Math.round(msg.value / minimalMoveTime)
    const num = Math.floor(time / 125) // wait_events=125
    time -= num * 125
    let ret = ''
    for (let i = 0; i < num; i++) {
      ret += 't125 '
    }
    if (time > 0) ret += `t${time} `
    return ret
  } else return ''
}

let dropRecordingTime = false

export const midiMessageHandler = async (data: MidiMessage) => {
  if (data.messageType === 'ControlChange') {
    commonStore.setInstrumentType(
      Math.round((data.value / 127) * (InstrumentTypeNameMap.length - 1))
    )
    return
  }
  if (commonStore.recordingTrackId) {
    if (dropRecordingTime && data.messageType === 'ElapsedTime') {
      dropRecordingTime = false
      return
    }
    data = {
      ...data,
      instrument: commonStore.instrumentType,
    }
    commonStore.setRecordingRawContent([
      ...commonStore.recordingRawContent,
      data,
    ])
    commonStore.setRecordingContent(
      commonStore.recordingContent + midiMessageToToken(data)
    )

    //TODO data.channel = data.instrument;
    PlayNote(data)
  }
}

type TrackProps = {
  id: string
  right: number
  scale: number
  isSelected: boolean
  onSelect: (id: string) => void
}

const Track: React.FC<TrackProps> = observer(
  ({ id, right, scale, isSelected, onSelect }) => {
    const { t } = useTranslation()
    const trackIndex = commonStore.tracks.findIndex((t) => t.id === id)!
    const track = commonStore.tracks[trackIndex]
    const trackClass = isSelected
      ? 'bg-blue-600'
      : commonStore.settings.darkMode
        ? 'bg-blue-900'
        : 'bg-gray-700'
    const controlX = useRef(0)

    let trackName = t('Track') + ' ' + id
    if (track.mainInstrument)
      trackName =
        t('Track') +
        ' - ' +
        t('Piano is the main instrument')!.replace(
          t('Piano')!,
          t(track.mainInstrument)
        ) +
        (track.content && ' - ' + track.content)
    else if (track.content) trackName = t('Track') + ' - ' + track.content

    return (
      <Draggable
        axis="x"
        bounds={{ left: 0, right }}
        grid={[snapValue, snapValue]}
        position={{
          x:
            ((track.offsetTime - commonStore.trackCurrentTime) /
              (baseMoveTime * scale)) *
            snapValue,
          y: 0,
        }}
        onStart={(e, data) => {
          controlX.current = data.lastX
        }}
        onStop={(e, data) => {
          const delta = data.lastX - controlX.current
          let offsetTime =
            Math.round(
              Math.round((delta / snapValue) * baseMoveTime * scale) /
                minimalMoveTime
            ) * minimalMoveTime
          offsetTime = Math.min(
            Math.max(offsetTime, -track.offsetTime),
            commonStore.trackTotalTime - track.offsetTime
          )

          const tracks = commonStore.tracks.slice()
          tracks[trackIndex].offsetTime += offsetTime
          commonStore.setTracks(tracks)
          refreshTracksTotalTime()
        }}
      >
        <div
          className={`cursor-move overflow-hidden whitespace-nowrap rounded p-1 ${trackClass}`}
          style={{
            width: `${Math.max(
              minimalTrackWidth,
              (track.contentTime / (baseMoveTime * scale)) * snapValue
            )}px`,
          }}
          onClick={() => onSelect(id)}
        >
          <span className="text-white">{trackName}</span>
        </div>
      </Draggable>
    )
  }
)

const AudiotrackEditor: FC<{ setPrompt: (prompt: string) => void }> = observer(
  ({ setPrompt }) => {
    const { t } = useTranslation()

    const viewControlsContainerRef = useRef<HTMLDivElement>(null)
    const currentTimeControlRef = useRef<HTMLDivElement>(null)
    const playStartTimeControlRef = useRef<HTMLDivElement>(null)
    const tracksEndLineRef = useRef<HTMLDivElement>(null)
    const tracksRef = useRef<HTMLDivElement>(null)
    const toolbarRef = useRef<HTMLDivElement>(null)
    const toolbarButtonRef = useRef<HTMLDivElement>(null)
    const toolbarSliderRef = useRef<HTMLInputElement>(null)
    const contentPreviewRef = useRef<HTMLDivElement>(null)

    const [refreshRef, setRefreshRef] = useState(false)

    const windowSize = useWindowSize()
    const scale = scaleMin + scaleMax - commonStore.trackScale

    const [selectedTrackId, setSelectedTrackId] = useState<string>('')
    const playStartTimeControlX = useRef(0)
    const selectedTrack = selectedTrackId
      ? commonStore.tracks.find((t) => t.id === selectedTrackId)
      : undefined

    useEffect(() => {
      if (toolbarSliderRef.current && toolbarSliderRef.current.parentElement)
        toolbarSliderRef.current.parentElement.style.removeProperty(
          '--fui-Slider--steps-percent'
        )
    }, [])

    const scrollContentToBottom = () => {
      if (contentPreviewRef.current)
        contentPreviewRef.current.scrollTop =
          contentPreviewRef.current.scrollHeight
    }

    useEffect(() => {
      scrollContentToBottom()
    }, [commonStore.recordingContent])

    useEffect(() => {
      setRefreshRef(!refreshRef)
    }, [windowSize, commonStore.tracks])

    const viewControlsContainerWidth =
      toolbarRef.current && toolbarButtonRef.current && toolbarSliderRef.current
        ? toolbarRef.current.clientWidth -
          toolbarButtonRef.current.clientWidth -
          toolbarSliderRef.current.clientWidth -
          16 // 16 = ml-2 mr-2
        : 0
    const tracksWidth = viewControlsContainerWidth
    const timeOfTracksWidth =
      Math.floor(tracksWidth / snapValue) * // number of moves
      baseMoveTime *
      scale
    const currentTimeControlWidth =
      timeOfTracksWidth < commonStore.trackTotalTime
        ? (timeOfTracksWidth / commonStore.trackTotalTime) *
          viewControlsContainerWidth
        : 0
    const playStartTimeControlPosition =
      ((commonStore.trackPlayStartTime - commonStore.trackCurrentTime) /
        (baseMoveTime * scale)) *
      snapValue
    const tracksEndPosition =
      ((commonStore.trackTotalTime - commonStore.trackCurrentTime) /
        (baseMoveTime * scale)) *
      snapValue
    const moveableTracksWidth =
      tracksEndLineRef.current &&
      viewControlsContainerRef.current &&
      tracksEndLineRef.current.getBoundingClientRect().left -
        (viewControlsContainerRef.current.getBoundingClientRect().left +
          trackInitOffsetPx) >
        0
        ? tracksEndLineRef.current.getBoundingClientRect().left -
          (viewControlsContainerRef.current.getBoundingClientRect().left +
            trackInitOffsetPx)
        : Infinity

    return (
      <div
        className="flex flex-col gap-2 overflow-hidden"
        style={{ width: '80vw', height: '80vh' }}
      >
        <div className="mx-auto">
          <Text
            size={100}
          >{`${commonStore.trackPlayStartTime} ms / ${commonStore.trackTotalTime} ms`}</Text>
        </div>
        <div className="flex border-b pb-2" ref={toolbarRef}>
          <div className="flex gap-2" ref={toolbarButtonRef}>
            <ToolTipButton
              disabled
              desc={t('Play All') + ' (Unavailable)'}
              icon={<Play16Regular />}
            />
            <ToolTipButton
              desc={t('Clear All')}
              icon={<Delete16Regular />}
              onClick={() => {
                commonStore.setTracks([])
                commonStore.setTrackScale(1)
                commonStore.setTrackTotalTime(tracksMinimalTotalTime)
                commonStore.setTrackCurrentTime(0)
                commonStore.setTrackPlayStartTime(0)
              }}
            />
          </div>
          <div className="grow">
            <div
              className="ml-2 mr-2 flex flex-col"
              ref={viewControlsContainerRef}
            >
              <div className="relative">
                <Tooltip
                  content={`${commonStore.trackTotalTime} ms`}
                  showDelay={0}
                  hideDelay={0}
                  relationship="description"
                >
                  <div
                    className="absolute border-l"
                    ref={tracksEndLineRef}
                    style={{
                      height:
                        tracksRef.current && commonStore.tracks.length > 0
                          ? tracksRef.current.clientHeight - arrowIconToTracks
                          : 0,
                      top: `${topToArrowIcon + arrowIconToTracks}px`,
                      left: `${tracksEndPosition + trackInitOffsetPx - pixelFix}px`,
                    }}
                  />
                </Tooltip>
              </div>
              <Draggable
                axis="x"
                bounds={{
                  left: 0,
                  right: viewControlsContainerWidth - currentTimeControlWidth,
                }}
                position={{
                  x:
                    (commonStore.trackCurrentTime /
                      commonStore.trackTotalTime) *
                    viewControlsContainerWidth,
                  y: 0,
                }}
                onDrag={(e, data) => {
                  setTimeout(() => {
                    let offset = 0
                    if (currentTimeControlRef.current) {
                      const match =
                        currentTimeControlRef.current.style.transform.match(
                          /translate\((.+)px,/
                        )
                      if (match) offset = parseFloat(match[1])
                    }
                    const offsetTime =
                      (commonStore.trackTotalTime /
                        viewControlsContainerWidth) *
                      offset
                    commonStore.setTrackCurrentTime(offsetTime)
                  }, 1)
                }}
              >
                <div
                  ref={currentTimeControlRef}
                  className={classnames(
                    'h-2 cursor-move rounded',
                    commonStore.settings.darkMode
                      ? 'bg-neutral-600'
                      : 'bg-gray-700'
                  )}
                  style={{ width: currentTimeControlWidth }}
                />
              </Draggable>
              <div
                className={classnames(
                  'flex',
                  (playStartTimeControlPosition < 0 ||
                    playStartTimeControlPosition >
                      viewControlsContainerWidth) &&
                    'hidden'
                )}
              >
                <Draggable
                  axis="x"
                  bounds={{
                    left: 0,
                    right: playStartTimeControlRef.current
                      ? Math.min(
                          viewControlsContainerWidth -
                            playStartTimeControlRef.current.clientWidth,
                          moveableTracksWidth
                        )
                      : 0,
                  }}
                  grid={[snapValue, snapValue]}
                  position={{ x: playStartTimeControlPosition, y: 0 }}
                  onStart={(e, data) => {
                    playStartTimeControlX.current = data.lastX
                  }}
                  onStop={(e, data) => {
                    const delta = data.lastX - playStartTimeControlX.current
                    let offsetTime =
                      Math.round(
                        Math.round((delta / snapValue) * baseMoveTime * scale) /
                          minimalMoveTime
                      ) * minimalMoveTime
                    offsetTime = Math.min(
                      Math.max(offsetTime, -commonStore.trackPlayStartTime),
                      commonStore.trackTotalTime -
                        commonStore.trackPlayStartTime
                    )
                    commonStore.setTrackPlayStartTime(
                      commonStore.trackPlayStartTime + offsetTime
                    )
                  }}
                >
                  <div
                    className="relative cursor-move"
                    ref={playStartTimeControlRef}
                  >
                    <ArrowAutofitWidth20Regular />
                    <div
                      className={classnames(
                        'absolute border-l',
                        commonStore.settings.darkMode
                          ? 'border-white'
                          : 'border-gray-700'
                      )}
                      style={{
                        height:
                          tracksRef.current && commonStore.tracks.length > 0
                            ? tracksRef.current.clientHeight
                            : 0,
                        top: '50%',
                        left: `calc(50% - ${pixelFix}px)`,
                      }}
                    />
                  </div>
                </Draggable>
              </div>
            </div>
          </div>
          <Tooltip
            content={t('Scale View')! + ': ' + commonStore.trackScale}
            showDelay={0}
            hideDelay={0}
            relationship="description"
          >
            <Slider
              ref={toolbarSliderRef}
              value={commonStore.trackScale}
              step={scaleMin}
              max={scaleMax}
              min={scaleMin}
              onChange={(e, data) => {
                commonStore.setTrackScale(data.value)
              }}
            />
          </Tooltip>
        </div>
        <div className="flex flex-col gap-1 overflow-y-auto" ref={tracksRef}>
          {commonStore.tracks.map((track) => (
            <div key={track.id} className="flex gap-2 border-b pb-1">
              <div className="flex h-7 gap-1 border-r">
                <ToolTipButton
                  desc={
                    commonStore.recordingTrackId === track.id
                      ? t('Stop')
                      : t('Record')
                  }
                  disabled={commonStore.platform === 'web'}
                  icon={
                    commonStore.recordingTrackId === track.id ? (
                      <Stop16Filled />
                    ) : (
                      <Record16Regular />
                    )
                  }
                  size="small"
                  shape="circular"
                  appearance="subtle"
                  onClick={() => {
                    flushMidiRecordingContent()
                    commonStore.setPlayingTrackId('')

                    if (commonStore.recordingTrackId === track.id) {
                      commonStore.setRecordingTrackId('')
                    } else {
                      if (commonStore.activeMidiDeviceIndex === -1) {
                        toast(t('Please select a MIDI device first'), {
                          type: 'warning',
                        })
                        return
                      }

                      dropRecordingTime = true
                      setSelectedTrackId(track.id)

                      commonStore.setRecordingTrackId(track.id)
                      commonStore.setRecordingContent(track.content)
                      commonStore.setRecordingRawContent(
                        track.rawContent.slice()
                      )
                    }
                  }}
                />
                <ToolTipButton
                  disabled
                  desc={
                    commonStore.playingTrackId === track.id
                      ? t('Stop')
                      : t('Play') + ' (Unavailable)'
                  }
                  icon={
                    commonStore.playingTrackId === track.id ? (
                      <Pause16Regular />
                    ) : (
                      <Play16Filled />
                    )
                  }
                  size="small"
                  shape="circular"
                  appearance="subtle"
                  onClick={() => {
                    flushMidiRecordingContent()
                    commonStore.setRecordingTrackId('')

                    if (commonStore.playingTrackId === track.id) {
                      commonStore.setPlayingTrackId('')
                    } else {
                      setSelectedTrackId(track.id)

                      commonStore.setPlayingTrackId(track.id)
                    }
                  }}
                />
                <ToolTipButton
                  desc={t('Delete')}
                  icon={<Delete16Regular />}
                  size="small"
                  shape="circular"
                  appearance="subtle"
                  onClick={() => {
                    const tracks = commonStore.tracks
                      .slice()
                      .filter((t) => t.id !== track.id)
                    commonStore.setTracks(tracks)
                    refreshTracksTotalTime()
                  }}
                />
              </div>
              <div className="relative grow overflow-hidden">
                <div className="absolute" style={{ left: -0 }}>
                  <Track
                    id={track.id}
                    scale={scale}
                    right={Math.min(tracksWidth, moveableTracksWidth)}
                    isSelected={selectedTrackId === track.id}
                    onSelect={setSelectedTrackId}
                  />
                </div>
              </div>
            </div>
          ))}
          <div className="flex items-center justify-between">
            <div className="flex gap-1">
              <Button
                icon={<Add16Regular />}
                size="small"
                shape="circular"
                appearance="subtle"
                disabled={commonStore.platform === 'web'}
                onClick={() => {
                  commonStore.setTracks([
                    ...commonStore.tracks,
                    {
                      id: uuid(),
                      mainInstrument: '',
                      content: '',
                      rawContent: [],
                      offsetTime: 0,
                      contentTime: 0,
                    },
                  ])
                }}
              >
                {t('New Track')}
              </Button>
              <Button
                icon={<ArrowUpload16Regular />}
                size="small"
                shape="circular"
                appearance="subtle"
                onClick={() => {
                  if (
                    commonStore.status.status === ModelStatus.Offline &&
                    !commonStore.settings.apiUrl &&
                    commonStore.platform !== 'web'
                  ) {
                    toast(
                      t(
                        'Please click the button in the top right corner to start the model'
                      ),
                      { type: 'warning' }
                    )
                    return
                  }

                  OpenFileDialog('*.mid').then(async (blob) => {
                    const bodyForm = new FormData()
                    bodyForm.append('file_data', blob)
                    const { url, headers } = await getReqUrl(
                      commonStore.getCurrentModelConfig().apiParameters.apiPort,
                      '/midi-to-text'
                    )
                    fetch(url, {
                      method: 'POST',
                      headers,
                      body: bodyForm,
                    })
                      .then(async (r) => {
                        if (r.status === 200) {
                          const text = (await r.json()).text as string
                          const rawContent = text
                            .split(' ')
                            .map(tokenToMidiMessage)
                            .filter((m) => m) as MidiMessage[]
                          const tracks = commonStore.tracks.slice()

                          tracks.push({
                            id: uuid(),
                            mainInstrument:
                              getMidiRawContentMainInstrument(rawContent),
                            content: text,
                            rawContent: rawContent,
                            offsetTime: 0,
                            contentTime: getMidiRawContentTime(rawContent),
                          })
                          commonStore.setTracks(tracks)
                          refreshTracksTotalTime()
                        } else {
                          toast(
                            'Failed to fetch - ' +
                              r.status +
                              ' - ' +
                              r.statusText +
                              ' - ' +
                              (await r.text()),
                            {
                              type: 'error',
                            }
                          )
                        }
                      })
                      .catch((e) => {
                        toast(t('Error') + ' - ' + (e.message || e), {
                          type: 'error',
                          autoClose: 2500,
                        })
                      })
                  })
                }}
              >
                {t('Import MIDI')}
              </Button>
            </div>
            <Text size={100}>{t('Select a track to preview the content')}</Text>
          </div>
        </div>
        <div className="grow"></div>
        {selectedTrack && (
          <Card
            size="small"
            appearance="outline"
            style={{ minHeight: '150px', maxHeight: '200px' }}
          >
            <div className="flex flex-col gap-1 overflow-hidden">
              <Text
                size={100}
              >{`${t('Start Time')}: ${selectedTrack.offsetTime} ms`}</Text>
              <Text
                size={100}
              >{`${t('Content Duration')}: ${selectedTrack.contentTime} ms`}</Text>
              <div
                className="overflow-y-auto overflow-x-hidden"
                ref={contentPreviewRef}
              >
                {selectedTrackId === commonStore.recordingTrackId
                  ? commonStore.recordingContent
                  : selectedTrack.content}
              </div>
            </div>
          </Card>
        )}
        {commonStore.platform !== 'web' && (
          <div className="mx-auto flex items-end gap-2">
            {t('Current Instrument') + ':'}
            {InstrumentTypeNameMap.map((name, i) => (
              <Text
                key={name}
                style={{ whiteSpace: 'nowrap' }}
                className={
                  commonStore.instrumentType === i ? 'text-blue-600' : ''
                }
                weight={commonStore.instrumentType === i ? 'bold' : 'regular'}
                size={commonStore.instrumentType === i ? 300 : 100}
              >
                {t(name)}
              </Text>
            ))}
          </div>
        )}
        <DialogTrigger disableButtonEnhancement>
          <Button
            icon={<MusicNote220Regular />}
            style={{ minHeight: '32px' }}
            onClick={() => {
              flushMidiRecordingContent()
              commonStore.setRecordingTrackId('')
              commonStore.setPlayingTrackId('')

              const timestamp = []
              const sortedTracks = commonStore.tracks
                .slice()
                .sort((a, b) => a.offsetTime - b.offsetTime)
              for (const track of sortedTracks) {
                timestamp.push(track.offsetTime)
                let accContentTime = 0
                for (const msg of track.rawContent) {
                  if (msg.messageType === 'ElapsedTime') {
                    accContentTime += msg.value
                    timestamp.push(track.offsetTime + accContentTime)
                  }
                }
              }
              const sortedTimestamp = timestamp.slice().sort((a, b) => a - b)
              const globalMessages: MidiMessage[] = sortedTimestamp.reduce(
                (messages, current, i) => [
                  ...messages,
                  {
                    messageType: 'ElapsedTime',
                    value: current - (i === 0 ? 0 : sortedTimestamp[i - 1]),
                  } as MidiMessage,
                ],
                [] as MidiMessage[]
              )
              for (const track of sortedTracks) {
                let currentTime = track.offsetTime
                let accContentTime = 0
                for (const msg of track.rawContent) {
                  if (msg.messageType === 'ElapsedTime') {
                    accContentTime += msg.value
                    currentTime = track.offsetTime + accContentTime
                  } else if (
                    msg.messageType === 'NoteOn' ||
                    msg.messageType === 'NoteOff'
                  ) {
                    const insertIndex = sortedTimestamp.findIndex(
                      (t) => t >= currentTime
                    )
                    globalMessages.splice(insertIndex + 1, 0, msg)
                    sortedTimestamp.splice(insertIndex + 1, 0, 0) // placeholder
                  }
                }
              }
              const result = (
                '<pad> ' + globalMessages.map(midiMessageToToken).join('')
              ).trim()
              commonStore.setCompositionSubmittedPrompt(result)
              setPrompt(result)
            }}
          >
            {t('Save to generation area')}
          </Button>
        </DialogTrigger>
      </div>
    )
  }
)

export default AudiotrackEditor
