import { FC, ReactElement } from 'react';
import { Label, Tooltip } from '@fluentui/react-components';
import classnames from 'classnames';

export const Labeled: FC<{
  label: string;
  desc?: string | null,
  content: ReactElement,
  flex?: boolean,
  spaceBetween?: boolean,
  breakline?: boolean
}> = ({
  label,
  desc,
  content,
  flex,
  spaceBetween,
  breakline
}) => {
  return (
    <div className={classnames(
      !breakline ? 'items-center' : '',
      flex ? 'flex' : 'grid grid-cols-2',
      breakline ? 'flex-col' : '',
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
