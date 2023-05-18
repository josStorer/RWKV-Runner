import React, {FC, MouseEventHandler} from 'react';
import commonStore, {ModelStatus} from '../stores/commonStore';
import {StartServer} from '../../wailsjs/go/backend_golang/App';
import {Button} from '@fluentui/react-components';
import {observer} from 'mobx-react-lite';
import {exit, readRoot, switchModel, updateConfig} from '../apis';
import {toast} from 'react-toastify';
import manifest from '../../../manifest.json';
import {getStrategy} from '../utils';
import {useTranslation} from 'react-i18next';

const mainButtonText = {
  [ModelStatus.Offline]: 'Run',
  [ModelStatus.Starting]: 'Starting',
  [ModelStatus.Loading]: 'Loading',
  [ModelStatus.Working]: 'Stop'
};

const onClickMainButton = async () => {
  const modelConfig = commonStore.getCurrentModelConfig();
  const port = modelConfig.apiParameters.apiPort;

  if (commonStore.modelStatus === ModelStatus.Offline) {
    commonStore.setModelStatus(ModelStatus.Starting);
    StartServer(port);

    let timeoutCount = 6;
    let loading = false;
    const intervalId = setInterval(() => {
      readRoot()
        .then(r => {
          if (r.ok && !loading) {
            clearInterval(intervalId);
            commonStore.setModelStatus(ModelStatus.Loading);
            loading = true;
            toast('Loading Model', {type: 'info'});
            updateConfig({
              max_tokens: modelConfig.apiParameters.maxResponseToken,
              temperature: modelConfig.apiParameters.temperature,
              top_p: modelConfig.apiParameters.topP,
              presence_penalty: modelConfig.apiParameters.presencePenalty,
              frequency_penalty: modelConfig.apiParameters.frequencyPenalty
            });
            switchModel({
              model: `${manifest.localModelDir}/${modelConfig.modelParameters.modelName}`,
              strategy: getStrategy(modelConfig)
            }).then((r) => {
              if (r.ok) {
                commonStore.setModelStatus(ModelStatus.Working);
                toast('Startup Completed', {type: 'success'});
              } else if (r.status === 304) {
                toast('Loading Model', {type: 'info'});
              } else {
                commonStore.setModelStatus(ModelStatus.Offline);
                toast('Failed to switch model', {type: 'error'});
              }
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
    exit();
  }
};

export const RunButton: FC<{ onClickRun?: MouseEventHandler }> = observer(({onClickRun}) => {
  const {t} = useTranslation();

  return (
    <Button disabled={commonStore.modelStatus === ModelStatus.Starting} appearance="primary" size="large"
            onClick={async (e) => {
              if (commonStore.modelStatus === ModelStatus.Offline)
                await onClickRun?.(e);
              await onClickMainButton();
            }}>
      {t(mainButtonText[commonStore.modelStatus])}
    </Button>
  );
});
