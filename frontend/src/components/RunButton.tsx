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
import { getStrategy, getSupportedCustomCudaFile, saveCache, toastWithButton } from '../utils';
import { useTranslation } from 'react-i18next';
import { ToolTipButton } from './ToolTipButton';
import { Play16Regular, Stop16Regular } from '@fluentui/react-icons';
import { useNavigate } from 'react-router';
import { BrowserOpenURL, WindowShow } from '../../wailsjs/runtime/runtime';

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
    if (commonStore.status.status === ModelStatus.Offline) {
      commonStore.setStatus({ status: ModelStatus.Starting });

      const modelConfig = commonStore.getCurrentModelConfig();
      let modelName = '';
      let modelPath = '';
      if (modelConfig && modelConfig.modelParameters) {
        modelName = modelConfig.modelParameters.modelName;
        modelPath = `${commonStore.settings.customModelsPath}/${modelName}`;
      } else {
        toast(t('Model Config Exception'), { type: 'error' });
        commonStore.setStatus({ status: ModelStatus.Offline });
        return;
      }

      if (!commonStore.depComplete) {
        let depErrorMsg = '';
        await DepCheck(commonStore.settings.customPythonPath).catch((e) => {
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
            if (depErrorMsg.includes('vc_redist')) {
              toastWithButton(t('Microsoft Visual C++ Redistributable is not installed, would you like to download it?'), t('Download'), () => {
                BrowserOpenURL('https://aka.ms/vs/16/release/vc_redist.x64.exe');
              });
            } else {
              toast(depErrorMsg, { type: 'info', position: 'bottom-left' });
              if (commonStore.platform != 'linux')
                toastWithButton(t('Python dependencies are incomplete, would you like to install them?'), t('Install'), () => {
                  InstallPyDep(commonStore.settings.customPythonPath, commonStore.settings.cnMirror).catch((e) => {
                    const errMsg = e.message || e;
                    toast(t('Error') + ' - ' + errMsg, { type: 'error' });
                  });
                  setTimeout(WindowShow, 1000);
                }, {
                  autoClose: 8000
                });
              else
                toastWithButton(t('On Linux system, you must manually install python dependencies.'), t('Check'), () => {
                  BrowserOpenURL('https://github.com/josStorer/RWKV-Runner/blob/master/build/linux/Readme_Install.txt');
                });
            }
          } else {
            toast(depErrorMsg, { type: 'error' });
          }
        });
        if (depErrorMsg) {
          commonStore.setStatus({ status: ModelStatus.Offline });
          return;
        }
        commonStore.setDepComplete(true);
        if (commonStore.platform === 'windows')
          CopyFile('./backend-python/wkv_cuda_utils/wkv_cuda_model.py', './py310/Lib/site-packages/rwkv/model.py');
        saveCache();
      }

      const currentModelSource = commonStore.modelSourceList.find(item => item.name === modelName);

      const showDownloadPrompt = (promptInfo: string, downloadName: string) => {
        toastWithButton(promptInfo, t('Download'), () => {
          const downloadUrl = currentModelSource?.downloadUrl;
          if (downloadUrl) {
            toastWithButton(`${t('Downloading')} ${downloadName}`, t('Check'), () => {
                navigate({ pathname: '/downloads' });
              },
              { autoClose: 3000 });
            AddToDownloadList(modelPath, downloadUrl);
          } else {
            toast(t('Can not find download url'), { type: 'error' });
          }
        });
      };

      if (!await FileExists(modelPath)) {
        showDownloadPrompt(t('Model file not found'), modelName);
        commonStore.setStatus({ status: ModelStatus.Offline });
        return;
      } else if (!currentModelSource?.isLocal) {
        showDownloadPrompt(t('Model file download is not complete'), modelName);
        commonStore.setStatus({ status: ModelStatus.Offline });
        return;
      }

      const port = modelConfig.apiParameters.apiPort;

      await exit(1000).catch(() => {
      });
      StartServer(commonStore.settings.customPythonPath, port, commonStore.settings.host !== '127.0.0.1' ? '0.0.0.0' : '127.0.0.1').catch((e) => {
        const errMsg = e.message || e;
        if (errMsg.includes('path contains space'))
          toast(`${t('Error')} - ${t('File Path Cannot Contain Space')}`, { type: 'error' });
        else
          toast(t('Error') + ' - ' + errMsg, { type: 'error' });
      });
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
            commonStore.setStatus({ status: ModelStatus.Loading });
            toast(t('Loading Model'), { type: 'info' });
            updateConfig({
              max_tokens: modelConfig.apiParameters.maxResponseToken,
              temperature: modelConfig.apiParameters.temperature,
              top_p: modelConfig.apiParameters.topP,
              presence_penalty: modelConfig.apiParameters.presencePenalty,
              frequency_penalty: modelConfig.apiParameters.frequencyPenalty
            });

            let customCudaFile = '';
            if ((modelConfig.modelParameters.device === 'CUDA' || modelConfig.modelParameters.device === 'Custom')
              && modelConfig.modelParameters.useCustomCuda) {
              if (commonStore.platform === 'windows') {
                customCudaFile = getSupportedCustomCudaFile();
                if (customCudaFile) {
                  FileExists('./py310/Lib/site-packages/rwkv/model.py').then((exist) => {
                    // defensive measure. As Python has already been launched, will only take effect the next time it runs.
                    if (!exist) CopyFile('./backend-python/wkv_cuda_utils/wkv_cuda_model.py', './py310/Lib/site-packages/rwkv/model.py');
                  });
                  await CopyFile(customCudaFile, './py310/Lib/site-packages/rwkv/wkv_cuda.pyd').catch(() => {
                    FileExists('./py310/Lib/site-packages/rwkv/wkv_cuda.pyd').then((exist) => {
                      if (!exist) {
                        customCudaFile = '';
                        toast(t('Failed to copy custom cuda file'), { type: 'error' });
                      }
                    });
                  });
                } else
                  toast(t('Supported custom cuda file not found'), { type: 'warning' });
              } else {
                customCudaFile = 'any';
              }
            }

            switchModel({
              model: modelPath,
              strategy: getStrategy(modelConfig),
              customCuda: customCudaFile !== ''
            }).then(async (r) => {
              if (r.ok) {
                commonStore.setStatus({ status: ModelStatus.Working });
                toastWithButton(t('Startup Completed'), t('Chat'), () => {
                  navigate({ pathname: '/chat' });
                }, { type: 'success', autoClose: 3000 });
              } else if (r.status === 304) {
                toast(t('Loading Model'), { type: 'info' });
              } else {
                commonStore.setStatus({ status: ModelStatus.Offline });
                toast(t('Failed to switch model') + ' - ' + await r.text(), { type: 'error' });
              }
            }).catch((e) => {
              commonStore.setStatus({ status: ModelStatus.Offline });
              toast(t('Failed to switch model') + ' - ' + e.message || e, { type: 'error' });
            });
          }
        }).catch(() => {
          if (timeoutCount <= 0) {
            clearInterval(intervalId);
            commonStore.setStatus({ status: ModelStatus.Offline });
          }
        });

        timeoutCount--;
      }, 1000);
    } else {
      commonStore.setStatus({ status: ModelStatus.Offline });
      exit();
    }
  };

  const onClick = async (e: any) => {
    if (commonStore.status.status === ModelStatus.Offline)
      await onClickRun?.(e);
    await onClickMainButton();
  };

  return (iconMode ?
      <ToolTipButton disabled={commonStore.status.status === ModelStatus.Starting}
        icon={iconModeButtonIcon[commonStore.status.status]}
        desc={t(mainButtonText[commonStore.status.status])}
        size="small" onClick={onClick} />
      :
      <Button disabled={commonStore.status.status === ModelStatus.Starting} appearance="primary" size="large"
        onClick={onClick}>
        {t(mainButtonText[commonStore.status.status])}
      </Button>
  );
});
