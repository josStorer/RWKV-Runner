import { toast } from 'react-toastify';
import commonStore from '../stores/commonStore';
import { t } from 'i18next';
import { ConvertSafetensors, FileExists, GetPyError } from '../../wailsjs/go/backend_golang/App';
import { WindowShow } from '../../wailsjs/runtime';
import { ModelConfig } from '../types/configs';

export const convertToSt = async (selectedConfig: ModelConfig) => {
  const modelPath = `${commonStore.settings.customModelsPath}/${selectedConfig.modelParameters.modelName}`;
  if (await FileExists(modelPath)) {
    toast(t('Start Converting'), { autoClose: 2000, type: 'info' });
    const newModelPath = modelPath.replace(/\.pth$/, '.st');
    ConvertSafetensors(modelPath, newModelPath).then(async () => {
      if (!await FileExists(newModelPath)) {
        if (commonStore.platform === 'windows' || commonStore.platform === 'linux')
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