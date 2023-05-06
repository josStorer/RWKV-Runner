import {FC, ReactElement} from 'react';
import {Label, Tooltip} from '@fluentui/react-components';

export const Labeled: FC<{ label: string; desc?: string, content: ReactElement }> = ({label, desc, content}) => {
  return (
    <div className="grid grid-cols-2 items-center">
      {desc ?
        <Tooltip content={desc} showDelay={0} hideDelay={0} relationship="description">
          <Label>{label}</Label>
        </Tooltip> :
        <Label>{label}</Label>
      }
      {content}
    </div>
  );
};
