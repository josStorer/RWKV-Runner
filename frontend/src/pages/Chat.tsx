import React, {FC} from 'react';
import {Page} from '../components/Page';
import {PresenceBadge} from '@fluentui/react-components';
import {useTranslation} from 'react-i18next';

export const Chat: FC = () => {
  const {t} = useTranslation();

  return (
    <Page title={t('Chat')} content={
      <div className="flex flex-col gap-2 overflow-hidden">
        <PresenceBadge/>
      </div>
    }/>
  );
};
