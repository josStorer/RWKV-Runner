import { FC, useState } from 'react'
import { CheckIcon, CopyIcon } from '@primer/octicons-react'
import { useTranslation } from 'react-i18next'
import { ClipboardSetText } from '../../wailsjs/runtime'
import { ToolTipButton } from './ToolTipButton'

export const CopyButton: FC<{ content: string; showDelay?: number }> = ({
  content,
  showDelay = 0,
}) => {
  const { t } = useTranslation()
  const [copied, setCopied] = useState(false)

  const onClick = () => {
    ClipboardSetText(content)
      .then(() => setCopied(true))
      .then(() =>
        setTimeout(() => {
          setCopied(false)
        }, 600)
      )
  }

  return (
    <ToolTipButton
      desc={t('Copy')}
      size="small"
      appearance="subtle"
      showDelay={showDelay}
      icon={copied ? <CheckIcon /> : <CopyIcon />}
      onClick={onClick}
    />
  )
}
