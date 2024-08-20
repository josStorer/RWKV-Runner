import { FC, ReactElement } from 'react'
import { Card, Text } from '@fluentui/react-components'

export const Section: FC<{
  title: string
  desc?: string | null
  content: ReactElement
  outline?: boolean
}> = ({ title, desc, content, outline = true }) => {
  return (
    <Card size="small" appearance={outline ? 'outline' : 'subtle'}>
      <div className="flex flex-col gap-5">
        <div className="flex flex-col gap-1">
          <Text weight="medium">{title}</Text>
          {desc && <Text size={100}>{desc}</Text>}
        </div>
      </div>
      <div className="overflow-y-auto overflow-x-hidden p-1">{content}</div>
    </Card>
  )
}
