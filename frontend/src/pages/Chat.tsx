import React, {FC} from 'react';
import {Page} from '../components/Page';
import {PresenceBadge} from '@fluentui/react-components';

export const Chat: FC = () => {
  return (
    <Page title="Chat" content={
      <div className="flex flex-col gap-2 overflow-hidden">
        <PresenceBadge/>
      </div>
    }/>
  );
};
