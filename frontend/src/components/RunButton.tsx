import React, {FC, MouseEventHandler, ReactElement} from 'react';
import commonStore, {ModelStatus} from '../stores/commonStore';
import {AddToDownloadList, DepCheck, FileExists, InstallPyDep, StartServer} from '../../wailsjs/go/backend_golang/App';
import {Button} from '@fluentui/react-components';
import {observer} from 'mobx-react-lite';
import {exit, readRoot, switchModel, updateConfig} from '../apis';
import {toast} from 'react-toastify';
import manifest from '../../../manifest.json';
import {getStrategy, saveCache, toastWithButton} from '../utils';
import {useTranslation} from 'react-i18next';
import {ToolTipButton} from './ToolTipButton';
import {Play16Regular, Stop16Regular} from '@fluentui/react-icons';
import {useNavigate} from 'react-router';

const mainButtonText = {
  [ModelStatus.Offline]: 'Run',
  [ModelStatus.Starting]: 'Starting',
  [ModelStatus.Loading]: 'Loading',
  [ModelStatus.Working]: 'Stop'
};

const iconModeButtonIcon: { [modelStatus: number]: ReactElement } = {
  [ModelStatus.Offline]: <Play16Regular/>,
  [ModelStatus.Starting]: <Stop16Regular/>,
  [ModelStatus.Loading]: <Stop16Regular/>,
  [ModelStatus.Working]: <Stop16Regular/>
};

export const RunButton: FC<{ onClickRun?: MouseEventHandler, iconMode?: boolean }>
  = observer(({
                onClickRun,
                iconMode
              }) => {
  const {t} = useTranslation();
  const navigate = useNavigate();

  const onClickMainButton = async () => {
    const modelConfig = commonStore.getCurrentModelConfig();
    const modelName = modelConfig.modelParameters.modelName;
    const modelPath = `./${manifest.localModelDir}/${modelName}`;

    if (!commonStore.depComplete) {
      let depErrorMsg = '';
      await DepCheck().catch((e) => {
        depErrorMsg = e.message || e;
        if (depErrorMsg === 'python zip not found') {
          toastWithButton(t('Python target not found, would you like to download it?'), t('Download'), () => {
            toastWithButton(`${t('Downloading')} Python`, t('Check'), () => {
              navigate({pathname: '/downloads'});
            }, {autoClose: 3000});
            AddToDownloadList('python-3.10.11-embed-amd64.zip', 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip');
          });
        } else if (depErrorMsg.includes('DepCheck Error')) {
          toastWithButton(t('Python dependencies are incomplete, would you like to install them?'), t('Install'), () => {
            InstallPyDep();
          });
        } else {
          toast(depErrorMsg, {type: 'error'});
        }
      });
      if (depErrorMsg) {
        return;
      }
      commonStore.setDepComplete(true);
      saveCache();
    }

    if (!await FileExists(modelPath)) {
      toastWithButton(t('Model file not found'), t('Download'), () => {
        const downloadUrl = commonStore.modelSourceList.find(item => item.name === modelName)?.downloadUrl;
        if (downloadUrl) {
          toastWithButton(`${t('Downloading')} ${modelName}`, t('Check'), () => {
              navigate({pathname: '/downloads'});
            },
            {autoClose: 3000});
          AddToDownloadList(modelPath, downloadUrl);
        } else {
          toast(t('Can not find download url'), {type: 'error'});
        }
      });

      return;
    }

    const port = modelConfig.apiParameters.apiPort;

    if (commonStore.modelStatus === ModelStatus.Offline) {
      commonStore.setModelStatus(ModelStatus.Starting);
      await exit(1000).catch(() => {
      });
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
              toast(t('Loading Model'), {type: 'info'});
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
                  toast(t('Startup Completed'), {type: 'success'});
                } else if (r.status === 304) {
                  toast(t('Loading Model'), {type: 'info'});
                } else {
                  commonStore.setModelStatus(ModelStatus.Offline);
                  toast(t('Failed to switch model'), {type: 'error'});
                }
              }).catch(() => {
                commonStore.setModelStatus(ModelStatus.Offline);
                toast(t('Failed to switch model'), {type: 'error'});
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
      exit();
    }
  };

  const onClick = async (e: any) => {
    if (commonStore.modelStatus === ModelStatus.Offline)
      await onClickRun?.(e);
    await onClickMainButton();
  };

  return (iconMode ?
      <ToolTipButton disabled={commonStore.modelStatus === ModelStatus.Starting}
                     icon={iconModeButtonIcon[commonStore.modelStatus]}
                     desc={t(mainButtonText[commonStore.modelStatus])}
                     size="small" onClick={onClick}/>
      :
      <Button disabled={commonStore.modelStatus === ModelStatus.Starting} appearance="primary" size="large"
              onClick={onClick}>
        {t(mainButtonText[commonStore.modelStatus])}
      </Button>
  );
});
