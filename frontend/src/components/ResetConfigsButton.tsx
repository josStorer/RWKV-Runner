import React, { FC } from 'react';
import { DialogButton } from './DialogButton';
import { useTranslation } from 'react-i18next';
import { ArrowReset20Regular } from '@fluentui/react-icons';
import commonStore from '../stores/commonStore';

import { defaultModelConfigs, defaultModelConfigsMac } from '../pages/defaultModelConfigs';

export const ResetConfigsButton: FC<{ afterConfirm?: () => void }> = ({ afterConfirm }) => {
  const { t } = useTranslation();
  return <DialogButton icon={<ArrowReset20Regular />} tooltip={t('Reset All Configs')} title={t('Reset All Configs')}
    contentText={t('Are you sure you want to reset all configs? This will obtain the latest preset configs, but will override your custom configs and cannot be undone.')}
    onConfirm={() => {
      commonStore.setModelConfigs(commonStore.platform != 'darwin' ? defaultModelConfigs : defaultModelConfigsMac, false);
      commonStore.setCurrentConfigIndex(0, true);
      afterConfirm?.();
    }} />;
};