import commonStore, {ModelSourceItem} from './stores/commonStore';
import {ReadJson, SaveJson} from '../wailsjs/go/backend_golang/App';
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
    () => {
      cache = {models: manifest.models};
      SaveJson('cache.json', cache).catch(() => {
      });
    }
  );
  commonStore.setModelSourceList(cache.models);

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
  commonStore.setModelSourceList(cache.models);
  SaveJson('cache.json', cache).catch(() => {
  });
}