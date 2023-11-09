import {
  AddToDownloadList,
  DeleteFile,
  DepCheck,
  InstallPyDep,
  ListDirFiles,
  ReadFileInfo,
  ReadJson,
  SaveJson,
  UpdateApp
} from '../../wailsjs/go/backend_golang/App';
import manifest from '../../../manifest.json';
import commonStore, { ModelStatus } from '../stores/commonStore';
import { toast } from 'react-toastify';
import { t } from 'i18next';
import { ToastOptions } from 'react-toastify/dist/types';
import { Button } from '@fluentui/react-components';
import { BrowserOpenURL, EventsOff, EventsOn, WindowShow } from '../../wailsjs/runtime';
import { NavigateFunction } from 'react-router';
import { ModelConfig, ModelParameters } from '../types/configs';
import { DownloadStatus } from '../types/downloads';
import { ModelSourceItem } from '../types/models';
import { Language, Languages, SettingsType } from '../types/settings';
import { DataProcessParameters, LoraFinetuneParameters } from '../types/train';

export type Cache = {
  version: string
  models: ModelSourceItem[]
  depComplete: boolean
}

export type LocalConfig = {
  modelSourceManifestList: string
  currentModelConfigIndex: number
  modelConfigs: ModelConfig[]
  settings: SettingsType,
  dataProcessParams: DataProcessParameters,
  loraFinetuneParams: LoraFinetuneParameters
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

const modelSuffix = ['.pth', '.st', '.safetensors'];

export async function refreshLocalModels(cache: {
  models: ModelSourceItem[]
}, filter: boolean = true, initUnfinishedModels: boolean = false) {
  if (filter)
    cache.models = cache.models.filter(m => !m.isComplete); //TODO BUG cause local but in manifest files to be removed, so currently cache is disabled

  await ListDirFiles(commonStore.settings.customModelsPath).then((data) => {
    cache.models.push(...data.flatMap(d => {
      if (!d.isDir && modelSuffix.some((ext => d.name.endsWith(ext))))
        return [{
          name: d.name,
          size: d.size,
          lastUpdated: d.modTime,
          isComplete: true,
          isLocal: true
        }] as ModelSourceItem[];
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
        } // else is not complete local file
        cache.models[i].isLocal = true;
        cache.models[i].localSize = cache.models[j].size;
        cache.models.splice(j, 1);
        j--;
      }
    }
  }
  commonStore.setModelSourceList(cache.models);
  if (initUnfinishedModels)
    initLastUnfinishedModelDownloads();
  await saveCache().catch(() => {
  });
}

function initLastUnfinishedModelDownloads() {
  const list: DownloadStatus[] = [];
  commonStore.modelSourceList.forEach((item) => {
    if (item.isLocal && !item.isComplete) {
      list.push(
        {
          name: item.name,
          path: `${commonStore.settings.customModelsPath}/${item.name}`,
          url: item.downloadUrl!,
          transferred: item.localSize!,
          size: item.size,
          speed: 0,
          progress: item.localSize! / item.size * 100,
          downloading: false,
          done: false
        }
      );
    }
  });
  commonStore.setLastUnfinishedModelDownloads(list);
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
    return modelSuffix.some((ext => model.name.endsWith(ext)))
      && index === self.findIndex(
        m => m.name === model.name || (m.SHA256 && m.SHA256 === model.SHA256 && m.size === model.size));
  });
  commonStore.setModelSourceList(cache.models);
  await saveCache().catch(() => {
  });
}

export const refreshModels = async (readCache: boolean = false, initUnfinishedModels: boolean = false) => {
  const cache = await refreshBuiltInModels(readCache);
  await refreshLocalModels(cache, false, initUnfinishedModels);
  await refreshRemoteModels(cache);
};

export const getStrategy = (modelConfig: ModelConfig | undefined = undefined) => {
  let params: ModelParameters;
  if (modelConfig) params = modelConfig.modelParameters;
  else params = commonStore.getCurrentModelConfig().modelParameters;
  const modelName = params.modelName.toLowerCase();
  const avoidOverflow = params.precision !== 'fp32' && modelName.includes('world') && (modelName.includes('0.1b') || modelName.includes('0.4b') ||
    modelName.includes('1.5b') || modelName.includes('1b5'));
  let strategy = '';
  switch (params.device) {
    case 'CPU':
      if (avoidOverflow)
        strategy = 'cpu fp32 *1 -> ';
      strategy += 'cpu ';
      strategy += params.precision === 'int8' ? 'fp32i8' : 'fp32';
      break;
    case 'WebGPU':
      strategy += params.precision === 'int8' ? 'fp16i8' : 'fp16';
      break;
    case 'CUDA':
    case 'CUDA-Beta':
      if (avoidOverflow)
        strategy = params.useCustomCuda ? 'cuda fp16 *1 -> ' : 'cuda fp32 *1 -> ';
      strategy += 'cuda ';
      strategy += params.precision === 'fp16' ? 'fp16' : params.precision === 'int8' ? 'fp16i8' : 'fp32';
      if (params.storedLayers < params.maxStoredLayers)
        strategy += ` *${params.storedLayers}+`;
      break;
    case 'MPS':
      if (avoidOverflow)
        strategy = 'mps fp32 *1 -> ';
      strategy += 'mps ';
      strategy += params.precision === 'fp16' ? 'fp16' : params.precision === 'int8' ? 'fp16i8' : 'fp32';
      break;
    case 'Custom':
      strategy = params.customStrategy || '';
      break;
  }
  return strategy;
};

export const saveConfigs = async () => {
  const data: LocalConfig = {
    modelSourceManifestList: commonStore.modelSourceManifestList,
    currentModelConfigIndex: commonStore.currentModelConfigIndex,
    modelConfigs: commonStore.modelConfigs,
    settings: commonStore.settings,
    dataProcessParams: commonStore.dataProcessParams,
    loraFinetuneParams: commonStore.loraFinetuneParams
  };
  return SaveJson('config.json', data);
};

export const saveCache = async () => {
  const data: Cache = {
    version: manifest.version,
    models: commonStore.modelSourceList,
    depComplete: commonStore.depComplete
  };
  return SaveJson('cache.json', data);
};

export const savePresets = async () => {
  return SaveJson('presets.json', commonStore.presets);
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
    if (path)
      ReadFileInfo(path).then(info => {
        if (info.size === 0 && url)
          AddToDownloadList(path, url.replace('@master', '@v' + manifest.version));
      }).catch(() => {
        if (url)
          AddToDownloadList(path, url.replace('@master', '@v' + manifest.version));
      });
  });
}

export function forceDownloadProgramFiles() {
  manifest.programFiles.forEach(({ url, path }) => {
    if (path && url)
      AddToDownloadList(path, url.replace('@master', '@v' + manifest.version));
  });
}

export async function deleteDynamicProgramFiles() {
  let promises: Promise<void>[] = [];
  manifest.programFiles.forEach(({ path }) => {
    if ((path.endsWith('.py') && !path.includes('get-pip.py')) || path.includes('requirements') || path.endsWith('.pyd'))
      promises.push(DeleteFile(path));
  });
  return await Promise.allSettled(promises).catch(() => {
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

export function bytesToReadable(size: number) {
  if (size < 1024) return size + ' B';
  else if (size < 1024 * 1024) return bytesToKb(size) + ' KB';
  else if (size < 1024 * 1024 * 1024) return bytesToMb(size) + ' MB';
  else return bytesToGb(size) + ' GB';
}

export function getServerRoot(defaultLocalPort: number) {
  const customApiUrl = commonStore.settings.apiUrl.trim().replace(/\/$/, '');
  if (customApiUrl)
    return customApiUrl;
  if (commonStore.platform === 'web')
    return '';
  return `http://127.0.0.1:${defaultLocalPort}`;
}

export function absPathAsset(path: string) {
  if (commonStore.platform === 'web')
    return path;
  if ((path.length > 0 && path[0] === '/') ||
    (path.length > 1 && path[1] === ':')) {
    return '=>' + path;
  }
  return path;
}

export async function checkUpdate(notifyEvenLatest: boolean = false) {
  fetch(!commonStore.settings.giteeUpdatesSource ?
    'https://api.github.com/repos/josstorer/RWKV-Runner/releases/latest' :
    'https://gitee.com/api/v5/repos/josc146/RWKV-Runner/releases/latest'
  ).then((r) => {
      if (r.ok) {
        r.json().then((data) => {
          if (data.tag_name) {
            const versionTag = data.tag_name;
            if (versionTag.replace('v', '') > manifest.version) {
              const verifyUrl = !commonStore.settings.giteeUpdatesSource ?
                `https://api.github.com/repos/josstorer/RWKV-Runner/releases/tags/${versionTag}` :
                `https://gitee.com/api/v5/repos/josc146/RWKV-Runner/releases/tags/${versionTag}`;

              fetch(verifyUrl).then((r) => {
                if (r.ok) {
                  r.json().then((data) => {
                    if (data.assets && data.assets.length > 0) {
                      const asset = data.assets.find((a: any) => a.name.toLowerCase().includes(commonStore.platform.toLowerCase()));
                      if (asset) {
                        const updateUrl = !commonStore.settings.giteeUpdatesSource ?
                          `https://github.com/josStorer/RWKV-Runner/releases/download/${versionTag}/${asset.name}` :
                          `https://gitee.com/josc146/RWKV-Runner/releases/download/${versionTag}/${asset.name}`;
                        toastWithButton(t('New Version Available') + ': ' + versionTag, t('Update'), () => {
                          DeleteFile('cache.json');
                          const progressId = 'update_app';
                          const progressEvent = 'updateApp';
                          const updateProgress = (ds: DownloadStatus | null) => {
                            const content =
                              t('Downloading update, please wait. If it is not completed, please manually download the program from GitHub and replace the original program.')
                              + (ds ? ` (${ds.progress.toFixed(2)}%  ${bytesToReadable(ds.transferred)}/${bytesToReadable(ds.size)})` : '');
                            const options: ToastOptions = {
                              type: 'info',
                              position: 'bottom-left',
                              autoClose: false,
                              toastId: progressId,
                              hideProgressBar: false,
                              progress: ds ? ds.progress / 100 : 0
                            };
                            if (toast.isActive(progressId))
                              toast.update(progressId, {
                                render: content,
                                ...options
                              });
                            else
                              toast(content, options);
                          };
                          updateProgress(null);
                          EventsOn(progressEvent, updateProgress);
                          UpdateApp(updateUrl).then(() => {
                            toast(t('Update completed, please restart the program.'), {
                                type: 'success',
                                position: 'bottom-left',
                                autoClose: false
                              }
                            );
                          }).catch((e) => {
                            toast(t('Update Error') + ' - ' + (e.message || e), {
                              type: 'error',
                              position: 'bottom-left',
                              autoClose: false
                            });
                          }).finally(() => {
                            toast.dismiss(progressId);
                            EventsOff(progressEvent);
                          });
                        }, {
                          autoClose: false,
                          position: 'bottom-left'
                        });
                      }
                    }
                  });
                } else {
                  throw new Error('Verify response was not ok.');
                }
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
    toast(t('Updates Check Error') + ' - ' + (e.message || e), { type: 'error', position: 'bottom-left' });
  });
}

export const checkDependencies = async (navigate: NavigateFunction) => {
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
        if (depErrorMsg.includes('vc_redist') || depErrorMsg.includes('DLL load failed while importing')) {
          toastWithButton(t('Microsoft Visual C++ Redistributable is not installed, would you like to download it?'), t('Download'), () => {
            BrowserOpenURL('https://aka.ms/vs/16/release/vc_redist.x64.exe');
          });
        } else {
          toast(depErrorMsg, { type: 'info', position: 'bottom-left' });
          if (commonStore.platform !== 'linux')
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
      return false;
    }
    commonStore.setDepComplete(true);
  }
  return true;
};

export function toastWithButton(text: string, buttonText: string, onClickButton: () => void, options?: ToastOptions) {
  let triggered = false;
  const id = toast(
    <div className="flex flex-row items-center justify-between">
      <div>{text}</div>
      <Button appearance="primary" onClick={() => {
        if (!triggered) {
          triggered = true;
          toast.dismiss(id);
          onClickButton();
        }
      }}>{buttonText}</Button>
    </div>,
    {
      type: 'info',
      ...options
    });
  return id;
}

export function getSupportedCustomCudaFile(isBeta: boolean) {
  if ([' 10', ' 16', ' 20', ' 30', 'MX', 'Tesla P', 'Quadro P', 'NVIDIA P', 'TITAN X', 'TITAN RTX', 'RTX A',
    'Quadro RTX 4000', 'Quadro RTX 5000', 'Tesla T4', 'NVIDIA A10', 'NVIDIA A40'].some(v => commonStore.status.device_name.includes(v)))
    return isBeta ?
      './backend-python/wkv_cuda_utils/beta/wkv_cuda10_30.pyd' :
      './backend-python/wkv_cuda_utils/wkv_cuda10_30.pyd';
  else if ([' 40', 'RTX 5000 Ada', 'RTX 6000 Ada', 'RTX TITAN Ada', 'NVIDIA L40'].some(v => commonStore.status.device_name.includes(v)))
    return isBeta ?
      './backend-python/wkv_cuda_utils/beta/wkv_cuda40.pyd' :
      './backend-python/wkv_cuda_utils/wkv_cuda40.pyd';
  else
    return '';
}