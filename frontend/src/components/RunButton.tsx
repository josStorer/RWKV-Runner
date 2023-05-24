import React, { FC, MouseEventHandler, ReactElement } from 'react';
import commonStore, { ModelStatus } from '../stores/commonStore';
import {
  AddToDownloadList,
  CopyFile,
  DepCheck,
  FileExists,
  InstallPyDep,
  StartServer
} from '../../wailsjs/go/backend_golang/App';
import { Button } from '@fluentui/react-components';
import { observer } from 'mobx-react-lite';
import { exit, getStatus, readRoot, switchModel, updateConfig } from '../apis';
import { toast } from 'react-toastify';
import manifest from '../../../manifest.json';
import { getStrategy, getSupportedCustomCudaFile, saveCache, toastWithButton } from '../utils';
import { useTranslation } from 'react-i18next';
import { ToolTipButton } from './ToolTipButton';
import { Play16Regular, Stop16Regular } from '@fluentui/react-icons';
import { useNavigate } from 'react-router';
import { WindowShow } from '../../wailsjs/runtime/runtime';

const mainButtonText = {
  [ModelStatus.Offline]: 'Run',
  [ModelStatus.Starting]: 'Starting',
  [ModelStatus.Loading]: 'Loading',
  [ModelStatus.Working]: 'Stop'
};

const iconModeButtonIcon: { [modelStatus: number]: ReactElement } = {
  [ModelStatus.Offline]: <Play16Regular />,
  [ModelStatus.Starting]: <Stop16Regular />,
  [ModelStatus.Loading]: <Stop16Regular />,
  [ModelStatus.Working]: <Stop16Regular />
};

export const RunButton: FC<{ onClickRun?: MouseEventHandler, iconMode?: boolean }>
  = observer(({
  onClickRun,
  iconMode
}) => {
  const { t } = useTranslation();
  const navigate = useNavigate();

  const onClickMainButton = async () => {
    if (commonStore.status.modelStatus === ModelStatus.Offline) {
      commonStore.setStatus({ modelStatus: ModelStatus.Starting });

      const modelConfig = commonStore.getCurrentModelConfig();
      let modelName = '';
      let modelPath = '';
      if (modelConfig && modelConfig.modelParameters) {
        modelName = modelConfig.modelParameters.modelName;
        modelPath = `./${manifest.localModelDir}/${modelName}`;
      } else {
        toast(t('Model Config Exception'), { type: 'error' });
        commonStore.setStatus({ modelStatus: ModelStatus.Offline });
        return;
      }

      if (!commonStore.depComplete) {
        let depErrorMsg = '';
        await DepCheck().catch((e) => {
          depErrorMsg = e.message || e;
          WindowShow();
          if (depErrorMsg === 'python zip not found') {
            toastWithButton(t('Python target not found, would you like to download it?'), t('Download'), () => {
              toastWithButton(`${t('Downloading')} Python`, t('Check'), () => {
                navigate({ pathname: '/downloads' });
              }, { autoClose: 3000 });
              AddToDownloadList('python-3.10.11-embed-amd64.zip', 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip');
            });
          } else if (depErrorMsg.includes('DepCheck Error')) {
            toastWithButton(t('Python dependencies are incomplete, would you like to install them?'), t('Install'), () => {
              InstallPyDep(commonStore.settings.cnMirror);
              setTimeout(WindowShow, 1000);
            });
          } else {
            toast(depErrorMsg, { type: 'error' });
          }
        });
        if (depErrorMsg) {
          commonStore.setStatus({ modelStatus: ModelStatus.Offline });
          return;
        }
        commonStore.setDepComplete(true);
        CopyFile('./backend-python/wkv_cuda_utils/wkv_cuda_model.py', './py310/Lib/site-packages/rwkv/model.py');
        saveCache();
      }

      if (!await FileExists(modelPath)) {
        toastWithButton(t('Model file not found'), t('Download'), () => {
          const downloadUrl = commonStore.modelSourceList.find(item => item.name === modelName)?.downloadUrl;
          if (downloadUrl) {
            toastWithButton(`${t('Downloading')} ${modelName}`, t('Check'), () => {
                navigate({ pathname: '/downloads' });
              },
              { autoClose: 3000 });
            AddToDownloadList(modelPath, downloadUrl);
          } else {
            toast(t('Can not find download url'), { type: 'error' });
          }
        });

        commonStore.setStatus({ modelStatus: ModelStatus.Offline });
        return;
      }

      const port = modelConfig.apiParameters.apiPort;

      await exit(1000).catch(() => {
      });
      StartServer(port, commonStore.settings.host);
      setTimeout(WindowShow, 1000);

      let timeoutCount = 6;
      let loading = false;
      const intervalId = setInterval(() => {
        readRoot()
        .then(async r => {
          if (r.ok && !loading) {
            loading = true;
            clearInterval(intervalId);
            await getStatus().then(status => {
              if (status)
                commonStore.setStatus(status);
            });
            commonStore.setStatus({ modelStatus: ModelStatus.Loading });
            toast(t('Loading Model'), { type: 'info' });
            updateConfig({
              max_tokens: modelConfig.apiParameters.maxResponseToken,
              temperature: modelConfig.apiParameters.temperature,
              top_p: modelConfig.apiParameters.topP,
              presence_penalty: modelConfig.apiParameters.presencePenalty,
              frequency_penalty: modelConfig.apiParameters.frequencyPenalty
            });

            let customCudaFile = '';
            if (modelConfig.modelParameters.useCustomCuda) {
              customCudaFile = getSupportedCustomCudaFile();
              if (customCudaFile) {
                FileExists('./py310/Lib/site-packages/rwkv/model.py').then((exist) => {
                  // defensive measure. As Python has already been launched, will only take effect the next time it runs.
                  if (!exist) CopyFile('./backend-python/wkv_cuda_utils/wkv_cuda_model.py', './py310/Lib/site-packages/rwkv/model.py');
                });
                await CopyFile(customCudaFile, './py310/Lib/site-packages/rwkv/wkv_cuda.pyd').catch(() => {
                  customCudaFile = '';
                  toast(t('Failed to copy custom cuda file'), { type: 'error' });
                });
              } else
                toast(t('Supported custom cuda file not found'), { type: 'warning' });
            }

            switchModel({
              model: `${manifest.localModelDir}/${modelConfig.modelParameters.modelName}`,
              strategy: getStrategy(modelConfig),
              customCuda: customCudaFile !== ''
            }).then((r) => {
              if (r.ok) {
                commonStore.setStatus({ modelStatus: ModelStatus.Working });
                toastWithButton(t('Startup Completed'), t('Chat'), () => {
                  navigate({ pathname: '/chat' });
                }, { type: 'success', autoClose: 3000 });
              } else if (r.status === 304) {
                toast(t('Loading Model'), { type: 'info' });
              } else {
                commonStore.setStatus({ modelStatus: ModelStatus.Offline });
                toast(t('Failed to switch model'), { type: 'error' });
              }
            }).catch(() => {
              commonStore.setStatus({ modelStatus: ModelStatus.Offline });
              toast(t('Failed to switch model'), { type: 'error' });
            });
          }
        }).catch(() => {
          if (timeoutCount <= 0) {
            clearInterval(intervalId);
            commonStore.setStatus({ modelStatus: ModelStatus.Offline });
          }
        });

        timeoutCount--;
      }, 1000);
    } else {
      commonStore.setStatus({ modelStatus: ModelStatus.Offline });
      exit();
    }
  };

  const onClick = async (e: any) => {
    if (commonStore.status.modelStatus === ModelStatus.Offline)
      await onClickRun?.(e);
    await onClickMainButton();
  };

  return (iconMode ?
      <ToolTipButton disabled={commonStore.status.modelStatus === ModelStatus.Starting}
        icon={iconModeButtonIcon[commonStore.status.modelStatus]}
        desc={t(mainButtonText[commonStore.status.modelStatus])}
        size="small" onClick={onClick} />
      :
      <Button disabled={commonStore.status.modelStatus === ModelStatus.Starting} appearance="primary" size="large"
        onClick={onClick}>
        {t(mainButtonText[commonStore.status.modelStatus])}
      </Button>
  );
});
