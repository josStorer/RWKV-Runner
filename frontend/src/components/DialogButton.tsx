import React, { FC, ReactElement } from 'react'
import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  DialogTrigger,
} from '@fluentui/react-components'
import { useTranslation } from 'react-i18next'
import { LazyImportComponent } from './LazyImportComponent'
import { ToolTipButton } from './ToolTipButton'

const MarkdownRender = React.lazy(() => import('./MarkdownRender'))

export const DialogButton: FC<{
  text?: string | null
  icon?: ReactElement
  tooltip?: string | null
  className?: string
  title: string
  content?: string | ReactElement | null
  markdown?: boolean
  onConfirm?: () => void
  size?: 'small' | 'medium' | 'large'
  shape?: 'rounded' | 'circular' | 'square'
  appearance?: 'secondary' | 'primary' | 'outline' | 'subtle' | 'transparent'
  cancelButton?: boolean
  confirmButton?: boolean
  cancelButtonText?: string
  confirmButtonText?: string
}> = ({
  text,
  icon,
  tooltip,
  className,
  title,
  content,
  markdown,
  onConfirm,
  size,
  shape,
  appearance,
  cancelButton = true,
  confirmButton = true,
  cancelButtonText = 'Cancel',
  confirmButtonText = 'Confirm',
}) => {
  const { t } = useTranslation()

  return (
    <Dialog>
      <DialogTrigger disableButtonEnhancement>
        {tooltip ? (
          <ToolTipButton
            className={className}
            desc={tooltip}
            text={text}
            icon={icon}
            size={size}
            shape={shape}
            appearance={appearance}
          />
        ) : (
          <Button
            className={className}
            icon={icon}
            size={size}
            shape={shape}
            appearance={appearance}
          >
            {text}
          </Button>
        )}
      </DialogTrigger>
      <DialogSurface style={{ transform: 'unset' }}>
        <DialogBody>
          <DialogTitle>{title}</DialogTitle>
          <DialogContent>
            {markdown ? (
              <LazyImportComponent lazyChildren={MarkdownRender}>
                {content}
              </LazyImportComponent>
            ) : (
              content
            )}
          </DialogContent>
          <DialogActions>
            {cancelButton && (
              <DialogTrigger disableButtonEnhancement>
                <Button appearance="secondary">{t(cancelButtonText)}</Button>
              </DialogTrigger>
            )}
            {confirmButton && (
              <DialogTrigger disableButtonEnhancement>
                <Button appearance="primary" onClick={onConfirm}>
                  {t(confirmButtonText)}
                </Button>
              </DialogTrigger>
            )}
          </DialogActions>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
}
