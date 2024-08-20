import React, { CSSProperties, FC } from 'react'
import { Input } from '@fluentui/react-components'
import { SliderOnChangeData } from '@fluentui/react-slider'

export const NumberInput: FC<{
  value: number
  min: number
  max: number
  step?: number
  onChange?: (
    ev: React.ChangeEvent<HTMLInputElement>,
    data: SliderOnChangeData
  ) => void
  style?: CSSProperties
  toFixed?: number
  disabled?: boolean
}> = ({ value, min, max, step, onChange, style, toFixed = 2, disabled }) => {
  return (
    <Input
      type="number"
      style={style}
      value={value.toString()}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e, data) => {
        onChange?.(e, { value: Number(data.value) })
      }}
      onBlur={(e) => {
        if (onChange) {
          if (step) {
            const offset = (min > 0 ? min : 0) - (max < 0 ? max : 0)
            value = Number(
              (Math.round((value - offset) / step) * step + offset).toFixed(
                toFixed
              )
            ) // avoid precision issues
          }
          onChange(e, { value: Math.max(Math.min(value, max), min) })
        }
      }}
    />
  )
}
