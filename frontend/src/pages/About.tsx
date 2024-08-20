import React, { FC } from 'react'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import MarkdownRender from '../components/MarkdownRender'
import { Page } from '../components/Page'
import commonStore from '../stores/commonStore'

const About: FC = observer(() => {
  const { t } = useTranslation()
  const lang: string = commonStore.settings.language

  return (
    <Page
      title={t('About')}
      content={
        <div className="overflow-y-auto overflow-x-hidden p-1">
          <MarkdownRender>
            {lang in commonStore.about
              ? commonStore.about[lang]
              : commonStore.about['en']}
          </MarkdownRender>
        </div>
      }
    />
  )
})

export default About
