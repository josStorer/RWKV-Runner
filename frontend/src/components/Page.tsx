import React, { FC, ReactElement } from 'react';
import { Divider, Text } from '@fluentui/react-components';

export const Page: FC<{ title: string; content: ReactElement }> = ({ title, content }) => {
  return (
    <div className="flex flex-col gap-2 p-2 h-full">
      <Text size={600}>{title}</Text>
      <Divider style={{ flexGrow: 0 }} />
      {content}
    </div>
  );
};
