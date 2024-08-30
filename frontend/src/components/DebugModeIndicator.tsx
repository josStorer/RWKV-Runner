import { FC } from 'react'
import classNames from 'classnames'

export const DebugModeIndicator: FC<{ showInDebugMode?: boolean }> = ({
  showInDebugMode = true,
}) => {
  if (import.meta.env.PROD) return <></>
  if (!showInDebugMode) return <></>
  return (
    <div
      className={classNames(
        'absolute',
        'right-1',
        'top-1',
        'p-1',
        'rounded',
        'bg-red-600',
        'bg-opacity-50',
        'text-white',
        'font-semibold',
        'text-opacity-50',
        'text-xs'
      )}
    >
      Debug Mode
    </div>
  )
}
