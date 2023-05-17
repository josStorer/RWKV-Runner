import commonStore, {ModelStatus} from '../stores/commonStore';

export const readRoot = async () => {
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;
  return fetch(`http://127.0.0.1:${port}`);
};

export const exit = async () => {
  commonStore.setModelStatus(ModelStatus.Offline);
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;
  return fetch(`http://127.0.0.1:${port}/exit`, {method: 'POST'});
};

export const switchModel = async (body: any) => {
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;
  return fetch(`http://127.0.0.1:${port}/switch-model`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
};

export const updateConfig = async (body: any) => {
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;
  return fetch(`http://127.0.0.1:${port}/update-config`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
};
