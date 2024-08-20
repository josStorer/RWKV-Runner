import React, {
  CSSProperties,
  FC,
  MouseEventHandler,
  ReactElement,
} from 'react'
import { Button, Tooltip } from '@fluentui/react-components'

export const ToolTipButton: FC<{
  text?: string | null
  desc: string
  icon?: ReactElement
  className?: string
  style?: CSSProperties
  size?: 'small' | 'medium' | 'large'
  shape?: 'rounded' | 'circular' | 'square'
  appearance?: 'secondary' | 'primary' | 'outline' | 'subtle' | 'transparent'
  disabled?: boolean
  onClick?: MouseEventHandler
  showDelay?: number
}> = ({
  text,
  desc,
  icon,
  className,
  style,
  size,
  shape,
  appearance,
  disabled,
  onClick,
  showDelay = 0,
}) => {
  return desc ? (
    <Tooltip
      content={desc}
      showDelay={showDelay}
      hideDelay={0}
      relationship="label"
    >
      <Button
        style={style}
        className={className}
        disabled={disabled}
        icon={icon}
        onClick={onClick}
        size={size}
        shape={shape}
        appearance={appearance}
      >
        {text}
      </Button>
    </Tooltip>
  ) : (
    <Button
      style={style}
      className={className}
      disabled={disabled}
      icon={icon}
      onClick={onClick}
      size={size}
      shape={shape}
      appearance={appearance}
    >
      {text}
    </Button>
  )
}
