import {Button, CompoundButton, Dropdown, Link, Option, Text} from '@fluentui/react-components';
import React, {FC, ReactElement} from 'react';
import Banner from '../assets/images/banner.jpg';
import {
  Chat20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  Storage20Regular
} from '@fluentui/react-icons';
import {useNavigate} from 'react-router';
import {SaveConfig} from '../../wailsjs/go/backend_golang/App';

type NavCard = {
  label: string;
  desc: string;
  path: string;
  icon: ReactElement;
};

export const navCards: NavCard[] = [
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

export const Home: FC = () => {
  const [selectedConfig, setSelectedConfig] = React.useState('RWKV-3B-4G MEM');

  const navigate = useNavigate();

  const onClickNavCard = (path: string) => {
    navigate({pathname: path});
  };

  return (
    <div className="flex flex-col justify-between h-full">
      <img className="rounded-xl select-none" src={Banner}/>
      <div className="flex flex-col gap-2">
        <Text size={600} weight="medium">Introduction</Text>
        <Text size={300}>
          RWKV is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT
          transformer (parallelizable). And it's 100% attention-free. You only need the hidden state at position t to
          compute the state at position t+1. You can use the "GPT" mode to quickly compute the hidden state for the
          "RNN" mode.
          <br/>
          So it's combining the best of RNN and transformer - great performance, fast inference, saves VRAM, fast
          training, "infinite" ctx_len, and free sentence embedding (using the final hidden state).
        </Text>
      </div>
      <div className="flex justify-between">
        {navCards.map(({label, path, icon, desc}, index) => (
          <CompoundButton className="w-1/5" icon={icon} secondaryContent={desc} key={`${path}-${index}`} value={path}
                          size="large" onClick={() => onClickNavCard(path)}>
            {label}
          </CompoundButton>
        ))}
      </div>
      <div className="flex justify-between">
        <div className="flex gap-4 items-end">
          Version: 1.0.0
          <Link>Help</Link>
        </div>
        <div className="flex gap-3">
          <Dropdown placeholder="Config"
                    value={selectedConfig}
                    onOptionSelect={(_, data) => {
                      if (data.optionValue)
                        setSelectedConfig(data.optionValue);
                    }}>
            <Option id="item-1" key="item-1">
              RWKV-3B-4G MEM
            </Option>
            <Option id="item-2" key="item-2">
              Item 2
            </Option>
            <Option id="item-3" key="item-3">
              Item 3
            </Option>
            <Option id="item-4" key="item-4">
              Item 4
            </Option>
          </Dropdown>
          <Button appearance="primary" size="large" onClick={() => SaveConfig({a: 1234, b: 'test'})}>Run</Button>
        </div>
      </div>
    </div>
  );
};
