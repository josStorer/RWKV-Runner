import commonStore from '../stores/commonStore';

export const readRoot = async () => {
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;
  return fetch(`http://127.0.0.1:${port}`);
};

export const exit = async (timeout?: number) => {
  const controller = new AbortController();
  if (timeout)
    setTimeout(() => controller.abort(), timeout);

  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;
  return fetch(`http://127.0.0.1:${port}/exit`, {method: 'POST', signal: controller.signal});
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
