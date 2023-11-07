import React, { FC } from 'react';
import { observer } from 'mobx-react-lite';
import { Divider, PresenceBadge, Text } from '@fluentui/react-components';
import commonStore, { ModelStatus } from '../stores/commonStore';
import { ConfigSelector } from './ConfigSelector';
import { RunButton } from './RunButton';
import { PresenceBadgeStatus } from '@fluentui/react-badge';
import { useTranslation } from 'react-i18next';

const statusText = {
  [ModelStatus.Offline]: 'Offline',
  [ModelStatus.Starting]: 'Starting',
  [ModelStatus.Loading]: 'Loading',
  [ModelStatus.Working]: 'Working'
};

const badgeStatus: { [modelStatus: number]: PresenceBadgeStatus } = {
  [ModelStatus.Offline]: 'unknown',
  [ModelStatus.Starting]: 'away',
  [ModelStatus.Loading]: 'away',
  [ModelStatus.Working]: 'available'
};

export const WorkHeader: FC = observer(() => {
  const { t } = useTranslation();
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;

  return commonStore.platform === 'web' ?
    <div /> :
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <PresenceBadge status={badgeStatus[commonStore.status.status]} />
          <Text size={100}>{t('Model Status') + ': ' + t(statusText[commonStore.status.status])}</Text>
        </div>
        <div className="flex items-center gap-2">
          <ConfigSelector size="small" />
          <RunButton iconMode />
        </div>
      </div>
      <Text size={100}>
        {t('This tool\'s API is compatible with OpenAI API. It can be used with any ChatGPT tool you like. Go to the settings of some ChatGPT tool, replace the \'https://api.openai.com\' part in the API address with \'') + `http://127.0.0.1:${port}` + '\'.'}
      </Text>
      <Divider style={{ flexGrow: 0 }} />
    </div>
    ;
});