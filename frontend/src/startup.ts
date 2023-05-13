import commonStore, {ModelSourceItem} from './stores/commonStore';
import {ListDirFiles, ReadJson, SaveJson} from '../wailsjs/go/backend_golang/App';
import manifest from '../../manifest.json';

export async function startup() {
  initConfig();
}

type Cache = {
  models: ModelSourceItem[]
}

async function initConfig() {
  let cache: Cache = {models: []};
  await ReadJson('cache.json').then((cacheData: Cache) => {
    cache = cacheData;
  }).catch(
    async () => {
      cache = {models: manifest.models};
      await SaveJson('cache.json', cache).catch(() => {
      });
    }
  );
  // built-in
  commonStore.setModelSourceList(cache.models);

  await ListDirFiles(manifest.localModelPath).then((data) => {
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

      if (cache.models[i].name === cache.models[j].name && cache.models[i].size === cache.models[j].size) {
        if (cache.models[i].lastUpdatedMs! < cache.models[j].lastUpdatedMs!) {
          cache.models[i] = Object.assign({}, cache.models[i], cache.models[j]);
        } else {
          cache.models[i] = Object.assign({}, cache.models[j], cache.models[i]);
        }
        cache.models.splice(j, 1);
        j--;
      }
    }
  }
  // local files
  commonStore.setModelSourceList(cache.models);
  await SaveJson('cache.json', cache).catch(() => {
  });

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
    return model.name.endsWith('.pth') && index === self.findIndex(m => m.SHA256 === model.SHA256 && m.size === model.size);
  });
  // remote files
  commonStore.setModelSourceList(cache.models);
  await SaveJson('cache.json', cache).catch(() => {
  });
}