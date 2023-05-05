import {FC, ReactElement} from 'react';
import {Label, Tooltip} from '@fluentui/react-components';

export const Labeled: FC<{ label: string; desc?: string, content: ReactElement }> = ({label, desc, content}) => {
  return (
    <div className="flex items-center">
      {desc ?
        <Tooltip content={desc} showDelay={0} hideDelay={0} relationship="description">
          <Label className="w-44">{label}</Label>
        </Tooltip> :
        <Label className="w-44">{label}</Label>
      }
      {content}
    </div>
  );
};
