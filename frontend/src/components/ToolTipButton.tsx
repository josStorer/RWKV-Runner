import React, {FC, MouseEventHandler, ReactElement} from 'react';
import {Button, Tooltip} from '@fluentui/react-components';

export const ToolTipButton: FC<{
  text?: string, desc: string, icon?: ReactElement, onClick?: MouseEventHandler
}> = ({
        text,
        desc,
        icon,
        onClick
      }) => {
  return (
    <Tooltip content={desc} showDelay={0} hideDelay={0} relationship="label">
      <Button icon={icon} onClick={onClick}>{text}</Button>
    </Tooltip>
  );
};
