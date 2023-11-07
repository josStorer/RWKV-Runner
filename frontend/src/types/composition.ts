import { NoteSequence } from '@magenta/music/esm/protobuf';

export type CompositionParams = {
  prompt: string,
  maxResponseToken: number,
  temperature: number,
  topP: number,
  autoPlay: boolean,
  useLocalSoundFont: boolean,
  midi: ArrayBuffer | null,
  ns: NoteSequence | null
}