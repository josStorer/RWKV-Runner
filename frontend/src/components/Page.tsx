import React, { FC, ReactElement } from 'react'
import { Divider, Text } from '@fluentui/react-components'

export const Page: FC<{ title: string; content: ReactElement }> = ({
  title,
  content,
}) => {
  return (
    <div className="flex h-full flex-col gap-2 p-2">
      <Text size={600}>{title}</Text>
      <Divider style={{ flexGrow: 0 }} />
      {content}
    </div>
  )
}
