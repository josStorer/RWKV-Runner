import React, * as React_2 from 'react';
import {CSSProperties, FC} from 'react';
import {Input} from '@fluentui/react-components';
import {SliderOnChangeData} from '@fluentui/react-slider';

export const NumberInput: FC<{
  value: number,
  min: number,
  max: number,
  step?: number,
  onChange?: (ev: React_2.ChangeEvent<HTMLInputElement>, data: SliderOnChangeData) => void
  style?: CSSProperties
}> = ({value, min, max, step, onChange, style}) => {
  return (
    <Input type="number" style={style} value={value.toString()} min={min} max={max} step={step}
           onChange={(e, data) => {
             onChange?.(e, {value: Number(data.value)});
           }}
           onBlur={(e) => {
             if (onChange) {
               if (step) {
                 const offset = (min > 0 ? min : 0) - (max < 0 ? max : 0);
                 value = Number(((
                     Math.round((value - offset) / step) * step)
                   + offset)
                   .toFixed(2)); // avoid precision issues
               }
               onChange(e, {value: Math.max(Math.min(value, max), min)});
             }
           }}/>
  );
};
