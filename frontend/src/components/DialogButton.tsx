import { FC, ReactElement } from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  DialogTrigger
} from '@fluentui/react-components';
import { ToolTipButton } from './ToolTipButton';
import { useTranslation } from 'react-i18next';

export const DialogButton: FC<{
  icon: ReactElement,
  tooltip: string,
  title: string,
  contentText: string,
  onConfirm: () => void
}> = ({ tooltip, icon, title, contentText, onConfirm }) => {
  const { t } = useTranslation();

  return <Dialog>
    <DialogTrigger disableButtonEnhancement>
      <ToolTipButton desc={tooltip} icon={icon} />
    </DialogTrigger>
    <DialogSurface>
      <DialogBody>
        <DialogTitle>{title}</DialogTitle>
        <DialogContent>
          {contentText}
        </DialogContent>
        <DialogActions>
          <DialogTrigger disableButtonEnhancement>
            <Button appearance="secondary">{t('Cancel')}</Button>
          </DialogTrigger>
          <DialogTrigger disableButtonEnhancement>
            <Button appearance="primary" onClick={onConfirm}>{t('Confirm')}
            </Button>
          </DialogTrigger>
        </DialogActions>
      </DialogBody>
    </DialogSurface>
  </Dialog>;
};