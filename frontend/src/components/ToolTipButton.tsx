import React, { FC, MouseEventHandler, ReactElement } from 'react';
import { Button, Tooltip } from '@fluentui/react-components';

export const ToolTipButton: FC<{
  text?: string | null,
  desc: string,
  icon?: ReactElement,
  className?: string,
  size?: 'small' | 'medium' | 'large',
  shape?: 'rounded' | 'circular' | 'square';
  appearance?: 'secondary' | 'primary' | 'outline' | 'subtle' | 'transparent';
  disabled?: boolean,
  onClick?: MouseEventHandler
}> = ({
  text,
  desc,
  icon,
  className,
  size,
  shape,
  appearance,
  disabled,
  onClick
}) => {
  return (
    <Tooltip content={desc} showDelay={0} hideDelay={0} relationship="label">
      <Button className={className} disabled={disabled} icon={icon} onClick={onClick} size={size} shape={shape}
        appearance={appearance}>{text}</Button>
    </Tooltip>
  );
};
