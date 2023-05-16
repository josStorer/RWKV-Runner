import React, {FC} from 'react';
import commonStore, {ModelStatus} from '../stores/commonStore';
import {StartServer} from '../../wailsjs/go/backend_golang/App';
import {Button} from '@fluentui/react-components';
import {observer} from 'mobx-react-lite';

const mainButtonText = {
  [ModelStatus.Offline]: 'Run',
  [ModelStatus.Starting]: 'Starting',
  [ModelStatus.Loading]: 'Loading',
  [ModelStatus.Working]: 'Stop'
};

const onClickMainButton = async () => {
  if (commonStore.modelStatus === ModelStatus.Offline) {
    commonStore.setModelStatus(ModelStatus.Starting);
    StartServer(commonStore.getStrategy(), `models\\${commonStore.getCurrentModelConfig().modelParameters.modelName}`);

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

export const RunButton: FC = observer(() => {
  return (
    <Button disabled={commonStore.modelStatus === ModelStatus.Starting} appearance="primary" size="large"
            onClick={onClickMainButton}>
      {mainButtonText[commonStore.modelStatus]}
    </Button>
  );
});
