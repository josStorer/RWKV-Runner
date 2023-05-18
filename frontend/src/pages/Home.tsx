import {CompoundButton, Dropdown, Link, Option, Text} from '@fluentui/react-components';
import React, {FC, ReactElement} from 'react';
import banner from '../assets/images/banner.jpg';
import {
  Chat20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  Storage20Regular
} from '@fluentui/react-icons';
import {useNavigate} from 'react-router';
import commonStore from '../stores/commonStore';
import {observer} from 'mobx-react-lite';
import {RunButton} from '../components/RunButton';
import manifest from '../../../manifest.json';
import {BrowserOpenURL} from '../../wailsjs/runtime';
import {useTranslation} from 'react-i18next';

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
    icon: <Chat20Regular/>
  },
  {
    label: 'Configs',
    desc: 'Manage your configs',
    path: '/configs',
    icon: <DocumentSettings20Regular/>
  },
  {
    label: 'Models',
    desc: 'Manage models',
    path: '/models',
    icon: <DataUsageSettings20Regular/>
  },
  {
    label: 'Train',
    desc: '',
    path: '/train',
    icon: <Storage20Regular/>
  }
];

export const Home: FC = observer(() => {
  const {t} = useTranslation();
  const navigate = useNavigate();

  const onClickNavCard = (path: string) => {
    navigate({pathname: path});
  };

  return (
    <div className="flex flex-col justify-between h-full">
      <img className="rounded-xl select-none hidden sm:block" src={banner}/>
      <div className="flex flex-col gap-2">
        <Text size={600} weight="medium">{t('Introduction')}</Text>
        <div className="h-40 overflow-y-auto p-1">
          {t('RWKV is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT transformer (parallelizable). And it\'s 100% attention-free. You only need the hidden state at position t to compute the state at position t+1. You can use the "GPT" mode to quickly compute the hidden state for the "RNN" mode. <br/> So it\'s combining the best of RNN and transformer - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding (using the final hidden state).')}
          {/*TODO Markdown*/}
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-5">
        {navCards.map(({label, path, icon, desc}, index) => (
          <CompoundButton icon={icon} secondaryContent={t(desc)} key={`${path}-${index}`} value={path}
                          size="large" onClick={() => onClickNavCard(path)}>
            {t(label)}
          </CompoundButton>
        ))}
      </div>
      <div className="flex flex-col gap-2">
        <div className="flex flex-row-reverse sm:fixed bottom-2 right-2">
          <div className="flex gap-3">
            <Dropdown style={{minWidth: 0}} listbox={{style: {minWidth: 0}}}
                      value={commonStore.getCurrentModelConfig().name}
                      selectedOptions={[commonStore.currentModelConfigIndex.toString()]}
                      onOptionSelect={(_, data) => {
                        if (data.optionValue)
                          commonStore.setCurrentConfigIndex(Number(data.optionValue));
                      }}>
              {commonStore.modelConfigs.map((config, index) =>
                <Option key={index} value={index.toString()}>{config.name}</Option>
              )}
            </Dropdown>
            <RunButton/>
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
