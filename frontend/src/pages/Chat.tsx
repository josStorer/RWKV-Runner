import React, {FC} from 'react';
import {useTranslation} from 'react-i18next';
import {RunButton} from '../components/RunButton';
import {Divider, PresenceBadge, Text} from '@fluentui/react-components';
import commonStore, {ModelStatus} from '../stores/commonStore';
import {observer} from 'mobx-react-lite';
import {PresenceBadgeStatus} from '@fluentui/react-badge';
import {ConfigSelector} from '../components/ConfigSelector';

const ChatPanel: FC = () => {
  return (
    <div></div>
  );
};

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

export const Chat: FC = observer(() => {
  const {t} = useTranslation();

  return (
    <div className="flex flex-col gap-1 p-2 h-full overflow-hidden">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <PresenceBadge status={badgeStatus[commonStore.modelStatus]}/>
          <Text size={100}>{t('Model Status') + ': ' + t(statusText[commonStore.modelStatus])}</Text>
        </div>
        <div className="flex items-center gap-2">
          <ConfigSelector size="small"/>
          <RunButton iconMode/>
        </div>
      </div>
      <Divider style={{flexGrow: 0}}/>
      <ChatPanel/>
    </div>
  );
});
