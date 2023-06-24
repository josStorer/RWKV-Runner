import commonStore, { Platform } from './stores/commonStore';
import { GetPlatform, ReadJson } from '../wailsjs/go/backend_golang/App';
import { Cache, checkUpdate, downloadProgramFiles, LocalConfig, refreshModels } from './utils';
import { getStatus } from './apis';
import { EventsOn } from '../wailsjs/runtime';
import manifest from '../../manifest.json';
import { defaultModelConfigs, defaultModelConfigsMac } from './pages/defaultModelConfigs';
import { Preset } from './pages/PresetsManager/PresetsButton';

export async function startup() {
  downloadProgramFiles();
  EventsOn('downloadList', (data) => {
    if (data)
      commonStore.setDownloadList(data);
  });

  initPresets();
  
  await GetPlatform().then(p => commonStore.setPlatform(p as Platform));
  await initConfig();

  initCache(true).then(initRemoteText); // depends on config customModelsPath

  if (commonStore.settings.autoUpdatesCheck) // depends on config settings
    checkUpdate();

  getStatus(1000).then(status => { // depends on config api port
    if (status)
      commonStore.setStatus(status);
  });
}

async function initRemoteText() {
  await fetch('https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner@master/manifest.json', { cache: 'no-cache' })
  .then(r => r.json()).then((data) => {
    if (data.version > manifest.version) {
      if (data.introduction)
        commonStore.setIntroduction(data.introduction);
      if (data.about)
        commonStore.setAbout(data.about);
    }
  });
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
    commonStore.setModelConfigs(commonStore.platform != 'darwin' ? defaultModelConfigs : defaultModelConfigsMac, true);
  });
}

async function initCache(initUnfinishedModels: boolean) {
  await ReadJson('cache.json').then((cacheData: Cache) => {
    if (cacheData.depComplete)
      commonStore.setDepComplete(cacheData.depComplete);
  }).catch(() => {
  });
  await refreshModels(false, initUnfinishedModels);
}

async function initPresets() {
  await ReadJson('presets.json').then((presets: Preset[]) => {
    commonStore.setPresets(presets, false);
  }).catch(() => {
  });
}
