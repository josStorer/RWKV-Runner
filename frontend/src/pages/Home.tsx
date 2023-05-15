import {Button, CompoundButton, Dropdown, Link, Option, Text} from '@fluentui/react-components';
import React, {FC, ReactElement} from 'react';
import banner from '../assets/images/banner.jpg';
import {
  Chat20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  Storage20Regular
} from '@fluentui/react-icons';
import {useNavigate} from 'react-router';
import commonStore, {ModelStatus} from '../stores/commonStore';
import {observer} from 'mobx-react-lite';
import {StartServer} from '../../wailsjs/go/backend_golang/App';

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

const mainButtonText = {
  [ModelStatus.Offline]: 'Run',
  [ModelStatus.Starting]: 'Starting',
  [ModelStatus.Loading]: 'Loading',
  [ModelStatus.Working]: 'Stop'
};

export const Home: FC = observer(() => {
  const [selectedConfig, setSelectedConfig] = React.useState('RWKV-3B-4G MEM');

  const navigate = useNavigate();

  const onClickNavCard = (path: string) => {
    navigate({pathname: path});
  };

  const onClickMainButton = async () => {
    if (commonStore.modelStatus === ModelStatus.Offline) {
      commonStore.setModelStatus(ModelStatus.Starting);
      StartServer('cuda fp16', 'models\\RWKV-4-Raven-1B5-v8-Eng-20230408-ctx4096.pth');

      let timeoutCount = 5;
      let loading = false;
      const intervalId = setInterval(() => {
        fetch('http://127.0.0.1:8000')
          .then(r => {
            if (r.ok && !loading) {
              clearInterval(intervalId);
              commonStore.setModelStatus(ModelStatus.Loading);
              loading = true;
              fetch('http://127.0.0.1:8000/update-config', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
              }).then(async (r) => {
                if (r.ok)
                  commonStore.setModelStatus(ModelStatus.Working);
              });
            }
          }).catch(() => {
          if (timeoutCount <= 0) {
            clearInterval(intervalId);
            commonStore.setModelStatus(ModelStatus.Offline);
          }
        });

        timeoutCount--;
      }, 1000);
    } else {
      commonStore.setModelStatus(ModelStatus.Offline);
      fetch('http://127.0.0.1:8000/exit', {method: 'POST'});
    }
  };

  return (
    <div className="flex flex-col justify-between h-full">
      <img className="rounded-xl select-none hidden sm:block" src={banner}/>
      <div className="flex flex-col gap-2">
        <Text size={600} weight="medium">Introduction</Text>
        <div className="h-40 overflow-y-auto p-1">
          RWKV is an RNN with Transformer-level LLM performance, which can also be directly trained like a GPT
          transformer (parallelizable). And it's 100% attention-free. You only need the hidden state at position t to
          compute the state at position t+1. You can use the "GPT" mode to quickly compute the hidden state for the
          "RNN" mode.
          <br/>
          So it's combining the best of RNN and transformer - great performance, fast inference, saves VRAM, fast
          training, "infinite" ctx_len, and free sentence embedding (using the final hidden state).
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-5">
        {navCards.map(({label, path, icon, desc}, index) => (
          <CompoundButton icon={icon} secondaryContent={desc} key={`${path}-${index}`} value={path}
                          size="large" onClick={() => onClickNavCard(path)}>
            {label}
          </CompoundButton>
        ))}
      </div>
      <div className="flex flex-col gap-2">
        <div className="flex flex-row-reverse sm:fixed bottom-2 right-2">
          <div className="flex gap-3">
            <Dropdown style={{minWidth: 0}}
                      placeholder="Config"
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
            <Button disabled={commonStore.modelStatus === ModelStatus.Starting} appearance="primary" size="large"
                    onClick={onClickMainButton}>
              {mainButtonText[commonStore.modelStatus]}
            </Button>
          </div>
        </div>
        <div className="flex gap-4 items-end">
          Version: 1.0.0
          <Link>Help</Link>
        </div>
      </div>
    </div>
  );
});
