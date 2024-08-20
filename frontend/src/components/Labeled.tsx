import { FC, ReactElement } from 'react'
import { Label, Tooltip } from '@fluentui/react-components'
import classnames from 'classnames'

export const Labeled: FC<{
  label: string
  desc?: string | null
  descComponent?: ReactElement
  content: ReactElement
  flex?: boolean
  spaceBetween?: boolean
  breakline?: boolean
  onMouseEnter?: () => void
  onMouseLeave?: () => void
}> = ({
  label,
  desc,
  descComponent,
  content,
  flex,
  spaceBetween,
  breakline,
  onMouseEnter,
  onMouseLeave,
}) => {
  return (
    <div
      className={classnames(
        !breakline ? 'items-center' : '',
        flex ? 'flex' : 'grid grid-cols-2',
        breakline ? 'flex-col' : '',
        spaceBetween && 'justify-between'
      )}
    >
      {desc || descComponent ? (
        <Tooltip
          content={descComponent ? descComponent : desc!}
          showDelay={0}
          hideDelay={0}
          relationship="description"
        >
          <Label onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
            {label}
          </Label>
        </Tooltip>
      ) : (
        <Label onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
          {label}
        </Label>
      )}
      {content}
    </div>
  )
}
