declare module JSX {
  import { PlayerElement } from 'html-midi-player'
  import { VisualizerElement } from 'html-midi-player'

  interface IntrinsicElements {
    'midi-player': PlayerElement
    'midi-visualizer': VisualizerElement
  }
}
