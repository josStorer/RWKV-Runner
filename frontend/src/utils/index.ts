import {ListDirFiles, ReadJson, SaveJson} from '../../wailsjs/go/backend_golang/App';
import manifest from '../../../manifest.json';
import commonStore, {ModelConfig, ModelParameters, ModelSourceItem} from '../stores/commonStore';

export type Cache = {
  models: ModelSourceItem[]
}

export type LocalConfig = {
  modelSourceManifestList: string
  currentModelConfigIndex: number
  modelConfigs: ModelConfig[]
}

export async function refreshBuiltInModels(readCache: boolean = false) {
  let cache: Cache = {models: []};
  if (readCache)
    await ReadJson('cache.json').then((cacheData: Cache) => {
      cache = cacheData;
    }).catch(
      async () => {
        cache = {models: manifest.models};
      }
    );
  else cache = {models: manifest.models};

  commonStore.setModelSourceList(cache.models);
  await saveCache().catch(() => {
  });
  return cache;
}

export async function refreshLocalModels(cache: Cache) {
  cache.models = cache.models.filter(m => !m.isLocal);

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
        if (cache.models[i].size === cache.models[j].size) {
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

export async function refreshRemoteModels(cache: Cache) {
  const manifestUrls = commonStore.modelSourceManifestList.split(/[,，;；\n]/);
  const requests = manifestUrls.filter(url => url.endsWith('.json')).map(
    url => fetch(url, {cache: 'no-cache'}).then(r => r.json()));

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
  strategy += (params.precision === 'fp16' ? 'fp16' : params.precision === 'int8' ? 'fp16i8' : 'fp32');
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
    modelConfigs: commonStore.modelConfigs
  };
  return SaveJson('config.json', data);
};

export const saveCache = async () => {
  const data: Cache = {
    models: commonStore.modelSourceList
  };
  return SaveJson('cache.json', data);
};