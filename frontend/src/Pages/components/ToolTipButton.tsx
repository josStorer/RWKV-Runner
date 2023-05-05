import React, {FC, ReactElement} from 'react';
import {Button, Tooltip} from '@fluentui/react-components';

export const ToolTipButton: FC<{ desc: string, icon: ReactElement }> = ({desc, icon}) => {
  return (
    <Tooltip content={desc} showDelay={0} hideDelay={0} relationship="label">
      <Button icon={icon}/>
    </Tooltip>
  );
};
