import { toast } from 'react-toastify';
import commonStore from '../stores/commonStore';
import { t } from 'i18next';
import { checkDependencies } from './index';
import { ConvertSafetensors, FileExists, GetPyError } from '../../wailsjs/go/backend_golang/App';
import { WindowShow } from '../../wailsjs/runtime';
import { NavigateFunction } from 'react-router';
import { ModelConfig } from '../types/configs';

export const convertToSt = async (navigate: NavigateFunction, selectedConfig: ModelConfig) => {
  if (commonStore.platform === 'linux') {
    toast(t('Linux is not yet supported for performing this operation, please do it manually.') + ' (backend-python/convert_safetensors.py)', { type: 'info' });
    return;
  }

  const ok = await checkDependencies(navigate);
  if (!ok)
    return;

  const modelPath = `${commonStore.settings.customModelsPath}/${selectedConfig.modelParameters.modelName}`;
  if (await FileExists(modelPath)) {
    toast(t('Start Converting'), { autoClose: 1000, type: 'info' });
    const newModelPath = modelPath.replace(/\.pth$/, '.st');
    ConvertSafetensors(commonStore.settings.customPythonPath, modelPath, newModelPath).then(async () => {
      if (!await FileExists(newModelPath)) {
        toast(t('Convert Failed') + ' - ' + await GetPyError(), { type: 'error' });
      } else {
        toast(`${t('Convert Success')} - ${newModelPath}`, { type: 'success' });
      }
    }).catch(e => {
      const errMsg = e.message || e;
      if (errMsg.includes('path contains space'))
        toast(`${t('Convert Failed')} - ${t('File Path Cannot Contain Space')}`, { type: 'error' });
      else
        toast(`${t('Convert Failed')} - ${e.message || e}`, { type: 'error' });
    });
    setTimeout(WindowShow, 1000);
  } else {
    toast(`${t('Model Not Found')} - ${modelPath}`, { type: 'error' });
  }
};