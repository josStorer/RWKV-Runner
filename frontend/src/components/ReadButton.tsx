import { FC, useState } from 'react'
import { MuteIcon, UnmuteIcon } from '@primer/octicons-react'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import commonStore from '../stores/commonStore'
import { ToolTipButton } from './ToolTipButton'

const synth = window.speechSynthesis

export const ReadButton: FC<{
  content: string
  inSpeaking?: boolean
  showDelay?: number
  setSpeakingOuter?: (speaking: boolean) => void
}> = observer(
  ({ content, inSpeaking = false, showDelay = 0, setSpeakingOuter }) => {
    const { t } = useTranslation()
    const [speaking, setSpeaking] = useState(inSpeaking)
    let lang: string = commonStore.settings.language
    if (lang === 'dev') lang = 'en'

    const setSpeakingInner = (speaking: boolean) => {
      setSpeakingOuter?.(speaking)
      setSpeaking(speaking)
    }

    const startSpeak = () => {
      synth.cancel()

      const utterance = new SpeechSynthesisUtterance(content)
      const voices = synth.getVoices()

      let voice
      if (lang === 'en')
        voice = voices.find((v) =>
          v.name.toLowerCase().includes('microsoft aria')
        )
      else if (lang === 'zh')
        voice = voices.find((v) => v.name.toLowerCase().includes('xiaoyi'))
      else if (lang === 'ja')
        voice = voices.find((v) => v.name.toLowerCase().includes('nanami'))
      if (!voice) voice = voices.find((v) => v.lang.substring(0, 2) === lang)
      if (!voice) voice = voices.find((v) => v.lang === navigator.language)

      Object.assign(utterance, {
        rate: 1,
        volume: 1,
        onend: () => setSpeakingInner(false),
        onerror: () => setSpeakingInner(false),
        voice: voice,
      })

      synth.speak(utterance)
      setSpeakingInner(true)
    }

    const stopSpeak = () => {
      synth.cancel()
      setSpeakingInner(false)
    }

    return (
      <ToolTipButton
        desc={t('Read Aloud')}
        size="small"
        appearance="subtle"
        showDelay={showDelay}
        icon={speaking ? <MuteIcon /> : <UnmuteIcon />}
        onClick={speaking ? stopSpeak : startSpeak}
      />
    )
  }
)
