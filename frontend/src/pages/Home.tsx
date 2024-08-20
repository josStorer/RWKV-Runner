import React, { FC } from 'react'
import { CompoundButton, Link, Text } from '@fluentui/react-components'
import {
  Chat20Regular,
  ClipboardEdit20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  MusicNote220Regular,
  Settings20Regular,
} from '@fluentui/react-icons'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router'
import manifest from '../../../manifest.json'
import { BrowserOpenURL } from '../../wailsjs/runtime'
import banner from '../assets/images/banner.jpg'
import { ConfigSelector } from '../components/ConfigSelector'
import { LazyImportComponent } from '../components/LazyImportComponent'
import { ResetConfigsButton } from '../components/ResetConfigsButton'
import { RunButton } from '../components/RunButton'
import commonStore from '../stores/commonStore'
import { NavCard } from '../types/home'
import { AdvancedGeneralSettings } from './Settings'

const clientNavCards: NavCard[] = [
  {
    label: 'Chat',
    desc: 'Go to chat page',
    path: '/chat',
    icon: <Chat20Regular />,
  },
  {
    label: 'Completion',
    desc: 'Writer, Translator, Role-playing',
    path: '/completion',
    icon: <ClipboardEdit20Regular />,
  },
  {
    label: 'Configs',
    desc: 'Manage your configs, adjust the starting model and parameters',
    path: '/configs',
    icon: <DocumentSettings20Regular />,
  },
  {
    label: 'Models',
    desc: 'Manage models',
    path: '/models',
    icon: <DataUsageSettings20Regular />,
  },
]

const webNavCards: NavCard[] = [
  {
    label: 'Chat',
    desc: 'Go to chat page',
    path: '/chat',
    icon: <Chat20Regular />,
  },
  {
    label: 'Completion',
    desc: 'Writer, Translator, Role-playing',
    path: '/completion',
    icon: <ClipboardEdit20Regular />,
  },
  {
    label: 'Composition',
    desc: '',
    path: '/composition',
    icon: <MusicNote220Regular />,
  },
  {
    label: 'Settings',
    desc: '',
    path: '/settings',
    icon: <Settings20Regular />,
  },
]

const MarkdownRender = React.lazy(() => import('../components/MarkdownRender'))

const Home: FC = observer(() => {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const lang: string = commonStore.settings.language

  const onClickNavCard = (path: string) => {
    navigate({ pathname: path })
  }

  return commonStore.platform === 'web' ? (
    <div className="flex h-full flex-col gap-2 overflow-y-auto overflow-x-hidden">
      <img
        className="grow select-none rounded-xl object-cover"
        style={{ maxHeight: '40%' }}
        src={banner}
      />
      <div className="grow"></div>
      <div className="grid grid-cols-2 gap-5 sm:grid-cols-4">
        {webNavCards.map(({ label, path, icon, desc }, index) => (
          <CompoundButton
            icon={icon}
            secondaryContent={t(desc)}
            key={`${path}-${index}`}
            value={path}
            size="large"
            onClick={() => onClickNavCard(path)}
          >
            {t(label)}
          </CompoundButton>
        ))}
      </div>
      <div className="flex flex-col gap-2">
        <AdvancedGeneralSettings />
        <div className="flex items-end gap-4">
          {t('Version')}: {manifest.version}
          <Link
            onClick={() =>
              BrowserOpenURL('https://github.com/josStorer/RWKV-Runner')
            }
          >
            {t('Help')}
          </Link>
        </div>
      </div>
    </div>
  ) : (
    <div className="flex h-full flex-col justify-between">
      <img
        className="hidden select-none rounded-xl object-cover sm:block"
        style={{ maxHeight: '40%' }}
        src={banner}
      />
      <div className="flex flex-col gap-2">
        <Text size={600} weight="medium">
          {t('Introduction')}
        </Text>
        <div className="h-40 overflow-y-auto overflow-x-hidden p-1">
          <LazyImportComponent lazyChildren={MarkdownRender}>
            {lang in commonStore.introduction
              ? commonStore.introduction[lang]
              : commonStore.introduction['en']}
          </LazyImportComponent>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-5 sm:grid-cols-4">
        {clientNavCards.map(({ label, path, icon, desc }, index) => (
          <CompoundButton
            icon={icon}
            secondaryContent={t(desc)}
            key={`${path}-${index}`}
            value={path}
            size="large"
            onClick={() => onClickNavCard(path)}
          >
            {t(label)}
          </CompoundButton>
        ))}
      </div>
      <div className="flex flex-col gap-2">
        <div className="bottom-2 right-2 flex flex-row-reverse sm:fixed">
          <div className="flex gap-3">
            <ResetConfigsButton />
            <ConfigSelector />
            <RunButton />
          </div>
        </div>
        <div className="flex items-end gap-4">
          {t('Version')}: {manifest.version}
          <Link
            onClick={() =>
              BrowserOpenURL('https://github.com/josStorer/RWKV-Runner')
            }
          >
            {t('Help')}
          </Link>
        </div>
      </div>
    </div>
  )
})

export default Home
