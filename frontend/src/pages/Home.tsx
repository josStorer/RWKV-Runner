import { CompoundButton, Link, Text } from '@fluentui/react-components';
import React, { FC } from 'react';
import banner from '../assets/images/banner.jpg';
import {
  Chat20Regular,
  ClipboardEdit20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  MusicNote220Regular,
  Settings20Regular
} from '@fluentui/react-icons';
import { useNavigate } from 'react-router';
import { observer } from 'mobx-react-lite';
import { RunButton } from '../components/RunButton';
import manifest from '../../../manifest.json';
import { BrowserOpenURL } from '../../wailsjs/runtime';
import { useTranslation } from 'react-i18next';
import { ConfigSelector } from '../components/ConfigSelector';
import commonStore from '../stores/commonStore';
import { ResetConfigsButton } from '../components/ResetConfigsButton';
import { AdvancedGeneralSettings } from './Settings';
import { NavCard } from '../types/home';
import { LazyImportComponent } from '../components/LazyImportComponent';

const clientNavCards: NavCard[] = [
  {
    label: 'Chat',
    desc: 'Go to chat page',
    path: '/chat',
    icon: <Chat20Regular />
  },
  {
    label: 'Completion',
    desc: 'Writer, Translator, Role-playing',
    path: '/completion',
    icon: <ClipboardEdit20Regular />
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
  }
];

const webNavCards: NavCard[] = [
  {
    label: 'Chat',
    desc: 'Go to chat page',
    path: '/chat',
    icon: <Chat20Regular />
  },
  {
    label: 'Completion',
    desc: 'Writer, Translator, Role-playing',
    path: '/completion',
    icon: <ClipboardEdit20Regular />
  },
  {
    label: 'Composition',
    desc: '',
    path: '/composition',
    icon: <MusicNote220Regular />
  },
  {
    label: 'Settings',
    desc: '',
    path: '/settings',
    icon: <Settings20Regular />
  }
];

const MarkdownRender = React.lazy(() => import('../components/MarkdownRender'));

const Home: FC = observer(() => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const lang: string = commonStore.settings.language;

  const onClickNavCard = (path: string) => {
    navigate({ pathname: path });
  };

  return commonStore.platform === 'web' ?
    (
      <div className="flex flex-col gap-2 h-full">
        <img className="rounded-xl select-none object-cover grow"
          style={{ maxHeight: '40%' }} src={banner} />
        <div className="grow"></div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-5">
          {webNavCards.map(({ label, path, icon, desc }, index) => (
            <CompoundButton icon={icon} secondaryContent={t(desc)} key={`${path}-${index}`} value={path}
              size="large" onClick={() => onClickNavCard(path)}>
              {t(label)}
            </CompoundButton>
          ))}
        </div>
        <div className="flex flex-col gap-2">
          <AdvancedGeneralSettings />
          <div className="flex gap-4 items-end">
            {t('Version')}: {manifest.version}
            <Link onClick={() => BrowserOpenURL('https://github.com/josStorer/RWKV-Runner')}>{t('Help')}</Link>
          </div>
        </div>
      </div>
    )
    : (
      <div className="flex flex-col justify-between h-full">
        <img className="rounded-xl select-none object-cover hidden sm:block"
          style={{ maxHeight: '40%' }} src={banner} />
        <div className="flex flex-col gap-2">
          <Text size={600} weight="medium">{t('Introduction')}</Text>
          <div className="h-40 overflow-y-auto overflow-x-hidden p-1">
            <LazyImportComponent lazyChildren={MarkdownRender}>
              {lang in commonStore.introduction ? commonStore.introduction[lang] : commonStore.introduction['en']}
            </LazyImportComponent>
          </div>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-5">
          {clientNavCards.map(({ label, path, icon, desc }, index) => (
            <CompoundButton icon={icon} secondaryContent={t(desc)} key={`${path}-${index}`} value={path}
              size="large" onClick={() => onClickNavCard(path)}>
              {t(label)}
            </CompoundButton>
          ))}
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex flex-row-reverse sm:fixed bottom-2 right-2">
            <div className="flex gap-3">
              <ResetConfigsButton />
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

export default Home;
