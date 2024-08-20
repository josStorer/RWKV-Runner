import { NoteSequence } from '@magenta/music/esm/protobuf'

export const tracksMinimalTotalTime = 5000

export type CompositionParams = {
  prompt: string
  maxResponseToken: number
  temperature: number
  topP: number
  autoPlay: boolean
  useLocalSoundFont: boolean
  externalPlay: boolean
  midi: ArrayBuffer | null
  ns: NoteSequence | null
  generationStartTime: number
  playOnlyGeneratedContent: boolean
}
export type Track = {
  id: string
  mainInstrument: string
  content: string
  rawContent: MidiMessage[]
  offsetTime: number
  contentTime: number
}
export type MidiPort = {
  name: string
}

export type MessageType = 'NoteOff' | 'NoteOn' | 'ElapsedTime' | 'ControlChange'

export type MidiMessage = {
  messageType: MessageType
  channel: number
  note: number
  velocity: number
  control: number
  value: number
  instrument: InstrumentType
}

export enum InstrumentType {
  Piano,
  Percussion,
  Drum,
  Tuba,
  Marimba,
  Bass,
  Guitar,
  Violin,
  Trumpet,
  Sax,
  Flute,
  Lead,
  Pad,
}

export const InstrumentTypeNameMap = [
  'Piano',
  'Percussion',
  'Drum',
  'Tuba',
  'Marimba',
  'Bass',
  'Guitar',
  'Violin',
  'Trumpet',
  'Sax',
  'Flute',
  'Lead',
  'Pad',
]

export const InstrumentTypeTokenMap = [
  'pi',
  'p',
  'd',
  't',
  'm',
  'b',
  'g',
  'v',
  'tr',
  's',
  'f',
  'l',
  'pa',
]
