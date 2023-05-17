import commonStore, {defaultModelConfigs} from './stores/commonStore';
import {ReadJson} from '../wailsjs/go/backend_golang/App';
import {LocalConfig, refreshModels} from './utils';

export async function startup() {
  initCache();
  await initConfig();
}

async function initConfig() {
  await ReadJson('config.json').then((configData: LocalConfig) => {
    if (configData.modelSourceManifestList)
      commonStore.setModelSourceManifestList(configData.modelSourceManifestList);

    if (configData.settings)
      commonStore.setSettings(configData.settings, false);

    if (configData.modelConfigs && Array.isArray(configData.modelConfigs))
      commonStore.setModelConfigs(configData.modelConfigs, false);
    else throw new Error('Invalid config.json');
    if (configData.currentModelConfigIndex &&
      configData.currentModelConfigIndex >= 0 && configData.currentModelConfigIndex < configData.modelConfigs.length)
      commonStore.setCurrentConfigIndex(configData.currentModelConfigIndex, false);
  }).catch(() => {
    commonStore.setModelConfigs(defaultModelConfigs, true);
  });
}

async function initCache() {
  await refreshModels(true);
}