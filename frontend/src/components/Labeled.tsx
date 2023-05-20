import {FC, ReactElement} from 'react';
import {Label, Tooltip} from '@fluentui/react-components';
import classnames from 'classnames';

export const Labeled: FC<{
  label: string; desc?: string | null, content: ReactElement, flex?: boolean, spaceBetween?: boolean
}> = ({
        label,
        desc,
        content,
        flex,
        spaceBetween
      }) => {
  return (
    <div className={classnames(
      'items-center',
      flex ? 'flex' : 'grid grid-cols-2',
      spaceBetween && 'justify-between')
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
