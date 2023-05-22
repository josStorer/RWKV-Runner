import React, { FC } from 'react';
import { useTranslation } from 'react-i18next';
import { Page } from '../components/Page';
import MarkdownRender from '../components/MarkdownRender';
import { observer } from 'mobx-react-lite';
import commonStore from '../stores/commonStore';

export type AboutContent = { [lang: string]: string }

export const About: FC = observer(() => {
  const { t } = useTranslation();
  const lang: string = commonStore.settings.language;

  return (
    <Page title={t('About')} content={
      <div className="overflow-y-auto overflow-x-hidden p-1">
        <MarkdownRender>
          {lang in commonStore.about ? commonStore.about[lang] : commonStore.about['en']}
        </MarkdownRender>
      </div>
    } />
  );
});
