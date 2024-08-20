import React, { FC, useEffect, useRef } from 'react'
import { Slider, Text } from '@fluentui/react-components'
import { SliderOnChangeData } from '@fluentui/react-slider'
import { NumberInput } from './NumberInput'

export const ValuedSlider: FC<{
  value: number
  min: number
  max: number
  step?: number
  input?: boolean
  onChange?: (
    ev: React.ChangeEvent<HTMLInputElement>,
    data: SliderOnChangeData
  ) => void
  toFixed?: number
  disabled?: boolean
}> = ({ value, min, max, step, input, onChange, toFixed, disabled }) => {
  const sliderRef = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (step && sliderRef.current && sliderRef.current.parentElement) {
      if ((max - min) / step > 10)
        sliderRef.current.parentElement.style.removeProperty(
          '--fui-Slider--steps-percent'
        )
    }
  }, [])

  return (
    <div className="flex items-center">
      <Slider
        ref={sliderRef}
        className="grow"
        style={{ minWidth: '50%' }}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={onChange}
        disabled={disabled}
      />
      {input ? (
        <NumberInput
          style={{ minWidth: 0 }}
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
          toFixed={toFixed}
          disabled={disabled}
        />
      ) : (
        <Text>{value}</Text>
      )}
    </div>
  )
}
