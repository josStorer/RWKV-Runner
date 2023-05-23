import commonStore from './stores/commonStore';
import { ReadJson } from '../wailsjs/go/backend_golang/App';
import { Cache, checkUpdate, downloadProgramFiles, LocalConfig, refreshModels, saveCache } from './utils';
import { getStatus } from './apis';
import { EventsOn } from '../wailsjs/runtime';
import { defaultModelConfigs } from './pages/Configs';

export async function startup() {
  downloadProgramFiles();
  EventsOn('downloadList', (data) => {
    if (data)
      commonStore.setDownloadList(data);
  });

  initCache().then(initRemoteText);

  await initConfig();

  if (commonStore.settings.autoUpdatesCheck) // depends on config settings
    checkUpdate();

  getStatus(500).then(status => { // depends on config api port
    if (status)
      commonStore.setStatus(status);
  });
}

async function initRemoteText() {
  await fetch('https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner/manifest.json', { cache: 'no-cache' })
  .then(r => r.json()).then((data) => {
    if (data.introduction)
      commonStore.setIntroduction(data.introduction);
    if (data.about)
      commonStore.setAbout(data.about);
  }).then(saveCache);
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
  await ReadJson('cache.json').then((cacheData: Cache) => {
    if (cacheData.introduction)
      commonStore.setIntroduction(cacheData.introduction);
    if (cacheData.about)
      commonStore.setAbout(cacheData.about);
    if (cacheData.depComplete)
      commonStore.setDepComplete(cacheData.depComplete);
  }).catch(() => {
  });
  await refreshModels(false);
}