import {FC, ReactElement} from 'react';
import {Text} from '@fluentui/react-components';

export const Section: FC<{ title: string; desc?: string, content: ReactElement }> = ({title, desc, content}) => {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <Text weight="medium">{title}</Text>
        {desc && <Text size={100}>{desc}</Text>}
      </div>
      {content}
    </div>
  );
};
