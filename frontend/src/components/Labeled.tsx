import {FC, ReactElement} from 'react';
import {Label, Tooltip} from '@fluentui/react-components';

export const Labeled: FC<{
  label: string; desc?: string, content: ReactElement, flex?: boolean, spaceBetween?: boolean
}> = ({
        label,
        desc,
        content,
        flex,
        spaceBetween
      }) => {
  return (
    <div className={
      (flex ? 'flex' : 'grid grid-cols-2') + ' ' +
      (spaceBetween ? 'justify-between' : '') + ' ' +
      'items-center'
    }>
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
