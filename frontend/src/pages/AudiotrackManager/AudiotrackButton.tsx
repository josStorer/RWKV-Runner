import React, { FC, lazy } from 'react';
import { useTranslation } from 'react-i18next';
import { Button, Dialog, DialogBody, DialogContent, DialogSurface, DialogTrigger } from '@fluentui/react-components';
import { CustomToastContainer } from '../../components/CustomToastContainer';
import { LazyImportComponent } from '../../components/LazyImportComponent';
import { flushMidiRecordingContent } from '../../utils';
import commonStore from '../../stores/commonStore';

const AudiotrackEditor = lazy(() => import('./AudiotrackEditor'));

export const AudiotrackButton: FC<{
  size?: 'small' | 'medium' | 'large',
  shape?: 'rounded' | 'circular' | 'square';
  appearance?: 'secondary' | 'primary' | 'outline' | 'subtle' | 'transparent';
}> = ({ size, shape, appearance }) => {
  const { t } = useTranslation();

  return <Dialog onOpenChange={(e, data) => {
    if (!data.open) {
      flushMidiRecordingContent();
      commonStore.setRecordingTrackId('');
    }
  }}>
    <DialogTrigger disableButtonEnhancement>
      <Button size={size} shape={shape} appearance={appearance}>
        {t('Open MIDI Input Audio Tracks')}
      </Button>
    </DialogTrigger>
    <DialogSurface style={{ paddingTop: 0, maxWidth: '90vw', width: 'fit-content' }}>
      <DialogBody>
        <DialogContent className="overflow-hidden">
          <CustomToastContainer />
          <LazyImportComponent lazyChildren={AudiotrackEditor} />
        </DialogContent>
      </DialogBody>
    </DialogSurface>
  </Dialog>;
};