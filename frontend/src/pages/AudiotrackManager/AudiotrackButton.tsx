import React, { FC, lazy } from 'react'
import {
  Button,
  Dialog,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTrigger,
} from '@fluentui/react-components'
import { useTranslation } from 'react-i18next'
import { CustomToastContainer } from '../../components/CustomToastContainer'
import { LazyImportComponent } from '../../components/LazyImportComponent'
import commonStore from '../../stores/commonStore'
import { flushMidiRecordingContent } from '../../utils'

const AudiotrackEditor = lazy(() => import('./AudiotrackEditor'))

export const AudiotrackButton: FC<{
  size?: 'small' | 'medium' | 'large'
  shape?: 'rounded' | 'circular' | 'square'
  appearance?: 'secondary' | 'primary' | 'outline' | 'subtle' | 'transparent'
  setPrompt: (prompt: string) => void
}> = ({ size, shape, appearance, setPrompt }) => {
  const { t } = useTranslation()

  return (
    <Dialog
      onOpenChange={(e, data) => {
        if (!data.open) {
          flushMidiRecordingContent()
          commonStore.setRecordingTrackId('')
          commonStore.setPlayingTrackId('')
        }
      }}
    >
      <DialogTrigger disableButtonEnhancement>
        <Button size={size} shape={shape} appearance={appearance}>
          {t('Open MIDI Input Audio Tracks')}
        </Button>
      </DialogTrigger>
      <DialogSurface
        style={{
          paddingTop: 0,
          maxWidth: '90vw',
          width: 'fit-content',
          transform: 'unset',
        }}
      >
        <DialogBody>
          <DialogContent className="overflow-hidden">
            <CustomToastContainer />
            <LazyImportComponent
              lazyChildren={AudiotrackEditor}
              lazyProps={{ setPrompt }}
            />
          </DialogContent>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
}
