import { CompoundButton, Link, Text } from '@fluentui/react-components';
import React, { FC, ReactElement } from 'react';
import banner from '../assets/images/banner.jpg';
import {
  Chat20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  Storage20Regular
} from '@fluentui/react-icons';
import { useNavigate } from 'react-router';
import { observer } from 'mobx-react-lite';
import { RunButton } from '../components/RunButton';
import manifest from '../../../manifest.json';
import { BrowserOpenURL } from '../../wailsjs/runtime';
import { useTranslation } from 'react-i18next';
import { ConfigSelector } from '../components/ConfigSelector';
import MarkdownRender from '../components/MarkdownRender';
import commonStore from '../stores/commonStore';

export type IntroductionContent = { [lang: string]: string }

type NavCard = {
  label: string;
  desc: string;
  path: string;
  icon: ReactElement;
};

const navCards: NavCard[] = [
  {
    label: 'Chat',
    desc: 'Go to chat page',
    path: '/chat',
    icon: <Chat20Regular />
  },
  {
    label: 'Configs',
    desc: 'Manage your configs',
    path: '/configs',
    icon: <DocumentSettings20Regular />
  },
  {
    label: 'Models',
    desc: 'Manage models',
    path: '/models',
    icon: <DataUsageSettings20Regular />
  },
  {
    label: 'Train',
    desc: '',
    path: '/train',
    icon: <Storage20Regular />
  }
];

export const Home: FC = observer(() => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const lang: string = commonStore.settings.language;

  const onClickNavCard = (path: string) => {
    navigate({ pathname: path });
  };

  return (
    <div className="flex flex-col justify-between h-full">
      <img className="rounded-xl select-none hidden sm:block" src={banner} />
      <div className="flex flex-col gap-2">
        <Text size={600} weight="medium">{t('Introduction')}</Text>
        <div className="h-40 overflow-y-auto overflow-x-hidden p-1">
          <MarkdownRender>
            {lang in commonStore.introduction ? commonStore.introduction[lang] : commonStore.introduction['en']}
          </MarkdownRender>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-5">
        {navCards.map(({ label, path, icon, desc }, index) => (
          <CompoundButton icon={icon} secondaryContent={t(desc)} key={`${path}-${index}`} value={path}
            size="large" onClick={() => onClickNavCard(path)}>
            {t(label)}
          </CompoundButton>
        ))}
      </div>
      <div className="flex flex-col gap-2">
        <div className="flex flex-row-reverse sm:fixed bottom-2 right-2">
          <div className="flex gap-3">
            <ConfigSelector />
            <RunButton />
          </div>
        </div>
        <div className="flex gap-4 items-end">
          {t('Version')}: {manifest.version}
          <Link onClick={() => BrowserOpenURL('https://github.com/josStorer/RWKV-Runner')}>{t('Help')}</Link>
        </div>
      </div>
    </div>
  );
});
