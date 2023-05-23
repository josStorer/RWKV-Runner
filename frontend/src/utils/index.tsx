import {
  AddToDownloadList,
  DeleteFile,
  DownloadFile,
  FileExists,
  ListDirFiles,
  ReadJson,
  SaveJson,
  UpdateApp
} from '../../wailsjs/go/backend_golang/App';
import manifest from '../../../manifest.json';
import commonStore from '../stores/commonStore';
import { toast } from 'react-toastify';
import { t } from 'i18next';
import { ToastOptions } from 'react-toastify/dist/types';
import { Button } from '@fluentui/react-components';
import { Language, Languages, SettingsType } from '../pages/Settings';
import { ModelSourceItem } from '../pages/Models';
import { ModelConfig, ModelParameters } from '../pages/Configs';
import { IntroductionContent } from '../pages/Home';
import { AboutContent } from '../pages/About';

export type Cache = {
  models: ModelSourceItem[]
  introduction: IntroductionContent,
  about: AboutContent
  depComplete: boolean
}

export type LocalConfig = {
  modelSourceManifestList: string
  currentModelConfigIndex: number
  modelConfigs: ModelConfig[]
  settings: SettingsType
}

export async function refreshBuiltInModels(readCache: boolean = false) {
  let cache: { models: ModelSourceItem[] } = { models: [] };
  if (readCache)
    await ReadJson('cache.json').then((cacheData: Cache) => {
      if (cacheData.models)
        cache.models = cacheData.models;
      else cache.models = manifest.models;
    }).catch(() => {
      cache.models = manifest.models;
    });
  else cache.models = manifest.models;

  commonStore.setModelSourceList(cache.models);
  await saveCache().catch(() => {
  });
  return cache;
}

export async function refreshLocalModels(cache: { models: ModelSourceItem[] }, filter: boolean = true) {
  if (filter)
    cache.models = cache.models.filter(m => !m.isLocal); //TODO BUG cause local but in manifest files to be removed, so currently cache is disabled

  await ListDirFiles(manifest.localModelDir).then((data) => {
    cache.models.push(...data.flatMap(d => {
      if (!d.isDir && d.name.endsWith('.pth'))
        return [{
          name: d.name,
          size: d.size,
          lastUpdated: d.modTime,
          isLocal: true
        }];
      return [];
    }));
  }).catch(() => {
  });

  for (let i = 0; i < cache.models.length; i++) {
    if (!cache.models[i].lastUpdatedMs)
      cache.models[i].lastUpdatedMs = Date.parse(cache.models[i].lastUpdated);

    for (let j = i + 1; j < cache.models.length; j++) {
      if (!cache.models[j].lastUpdatedMs)
        cache.models[j].lastUpdatedMs = Date.parse(cache.models[j].lastUpdated);

      if (cache.models[i].name === cache.models[j].name) {
        if (cache.models[i].size <= cache.models[j].size) { // j is local file
          if (cache.models[i].lastUpdatedMs! < cache.models[j].lastUpdatedMs!) {
            cache.models[i] = Object.assign({}, cache.models[i], cache.models[j]);
          } else {
            cache.models[i] = Object.assign({}, cache.models[j], cache.models[i]);
          }
        } // else is bad local file
        cache.models.splice(j, 1);
        j--;
      }
    }
  }
  commonStore.setModelSourceList(cache.models);
  await saveCache().catch(() => {
  });
}

export async function refreshRemoteModels(cache: { models: ModelSourceItem[] }) {
  const manifestUrls = commonStore.modelSourceManifestList.split(/[,，;；\n]/);
  const requests = manifestUrls.filter(url => url.endsWith('.json')).map(
    url => fetch(url, { cache: 'no-cache' }).then(r => r.json()));

  await Promise.allSettled(requests)
  .then((data: PromiseSettledResult<Cache>[]) => {
    cache.models.push(...data.flatMap(d => {
      if (d.status === 'fulfilled')
        return d.value.models;
      return [];
    }));
  })
  .catch(() => {
  });
  cache.models = cache.models.filter((model, index, self) => {
    return model.name.endsWith('.pth')
      && index === self.findIndex(
        m => m.name === model.name || (m.SHA256 === model.SHA256 && m.size === model.size));
  });
  commonStore.setModelSourceList(cache.models);
  await saveCache().catch(() => {
  });
}

export const refreshModels = async (readCache: boolean = false) => {
  const cache = await refreshBuiltInModels(readCache);
  await refreshLocalModels(cache);
  await refreshRemoteModels(cache);
};

export const getStrategy = (modelConfig: ModelConfig | undefined = undefined) => {
  let params: ModelParameters;
  if (modelConfig) params = modelConfig.modelParameters;
  else params = commonStore.getCurrentModelConfig().modelParameters;
  let strategy = '';
  strategy += (params.device === 'CPU' ? 'cpu' : 'cuda') + ' ';
  strategy += params.device === 'CPU' ? 'fp32' : (params.precision === 'fp16' ? 'fp16' : params.precision === 'int8' ? 'fp16i8' : 'fp32');
  if (params.storedLayers < params.maxStoredLayers)
    strategy += ` *${params.storedLayers}+`;
  if (params.enableHighPrecisionForLastLayer)
    strategy += ' -> cpu fp32 *1';
  return strategy;
};

export const saveConfigs = async () => {
  const data: LocalConfig = {
    modelSourceManifestList: commonStore.modelSourceManifestList,
    currentModelConfigIndex: commonStore.currentModelConfigIndex,
    modelConfigs: commonStore.modelConfigs,
    settings: commonStore.settings
  };
  return SaveJson('config.json', data);
};

export const saveCache = async () => {
  const data: Cache = {
    models: commonStore.modelSourceList,
    introduction: commonStore.introduction,
    about: commonStore.about,
    depComplete: commonStore.depComplete
  };
  return SaveJson('cache.json', data);
};

export function getUserLanguage(): Language {
  // const l = navigator.language.toLowerCase();
  // if (['zh-hk', 'zh-mo', 'zh-tw', 'zh-cht', 'zh-hant'].includes(l)) return 'zhHant'

  const l = navigator.language.substring(0, 2);
  if (l in Languages) return l as Language;
  return 'dev';
}

export function isSystemLightMode() {
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
}

export function downloadProgramFiles() {
  manifest.programFiles.forEach(({ url, path }) => {
    FileExists(path).then(exists => {
      if (!exists)
        AddToDownloadList(path, url);
    });
  });
}

export function forceDownloadProgramFiles() {
  manifest.programFiles.forEach(({ url, path }) => {
    DownloadFile(path, url);
  });
}

export function deleteDynamicProgramFiles() {
  DeleteFile('cache.json');
  manifest.programFiles.forEach(({ path }) => {
    if ((path.endsWith('.py') && !path.includes('get-pip.py')) || path.includes('requirements'))
      DeleteFile(path);
  });
}

export function bytesToGb(size: number) {
  return (size / 1024 / 1024 / 1024).toFixed(2);
}

export function bytesToMb(size: number) {
  return (size / 1024 / 1024).toFixed(2);
}

export function bytesToKb(size: number) {
  return (size / 1024).toFixed(2);
}

export async function checkUpdate(notifyEvenLatest: boolean = false) {
  let updateUrl = '';
  await fetch(!commonStore.settings.giteeUpdatesSource ?
    'https://api.github.com/repos/josstorer/RWKV-Runner/releases/latest' :
    'https://gitee.com/api/v5/repos/josc146/RWKV-Runner/releases/latest'
  ).then((r) => {
      if (r.ok) {
        r.json().then((data) => {
          if (data.tag_name) {
            const versionTag = data.tag_name;
            if (versionTag.replace('v', '') > manifest.version) {
              updateUrl = !commonStore.settings.giteeUpdatesSource ?
                `https://github.com/josStorer/RWKV-Runner/releases/download/${versionTag}/RWKV-Runner_windows_x64.exe` :
                `https://gitee.com/josc146/RWKV-Runner/releases/download/${versionTag}/RWKV-Runner_windows_x64.exe`;
              toastWithButton(t('New Version Available') + ': ' + versionTag, t('Update'), () => {
                deleteDynamicProgramFiles();
                setTimeout(() => {
                  UpdateApp(updateUrl).catch((e) => {
                    toast(t('Update Error, Please restart this program') + ' - ' + e.message || e, {
                      type: 'error',
                      position: 'bottom-left',
                      autoClose: false
                    });
                  });
                }, 500);
              }, {
                autoClose: false,
                position: 'bottom-left'
              });
            } else {
              if (notifyEvenLatest) {
                toast(t('This is the latest version'), { type: 'success', position: 'bottom-left', autoClose: 2000 });
              }
            }
          } else {
            throw new Error('Invalid response.');
          }
        });
      } else {
        throw new Error('Network response was not ok.');
      }
    }
  ).catch((e) => {
    toast(t('Updates Check Error') + ' - ' + e.message || e, { type: 'error', position: 'bottom-left' });
  });
  return updateUrl;
}

export function toastWithButton(text: string, buttonText: string, onClickButton: () => void, options?: ToastOptions) {
  return toast(
    <div className="flex flex-row items-center justify-between">
      <div>{text}</div>
      <Button appearance="primary" onClick={onClickButton}>{buttonText}</Button>
    </div>,
    {
      type: 'info',
      ...options
    });
}