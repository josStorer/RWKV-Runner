import React, {FC, MouseEventHandler, ReactElement} from 'react';
import {Button, Tooltip} from '@fluentui/react-components';

export const ToolTipButton: FC<{
  text?: string | null,
  desc: string,
  icon?: ReactElement,
  size?: 'small' | 'medium' | 'large',
  disabled?: boolean,
  onClick?: MouseEventHandler
}> = ({
        text,
        desc,
        icon,
        size,
        disabled,
        onClick
      }) => {
  return (
    <Tooltip content={desc} showDelay={0} hideDelay={0} relationship="label">
      <Button disabled={disabled} icon={icon} onClick={onClick} size={size}>{text}</Button>
    </Tooltip>
  );
};
