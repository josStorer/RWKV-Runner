import { TFunction } from 'i18next'
import { toast } from 'react-toastify'
import commonStore, { Status } from '../stores/commonStore'

export const readRoot = async () => {
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort
  return fetch(`http://127.0.0.1:${port}`)
}

export const exit = async (timeout?: number) => {
  const controller = new AbortController()
  if (timeout) setTimeout(() => controller.abort(), timeout)

  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort
  return fetch(`http://127.0.0.1:${port}/exit`, {
    method: 'POST',
    signal: controller.signal,
  })
}

export const switchModel = async (body: any) => {
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort
  return fetch(`http://127.0.0.1:${port}/switch-model`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
}

export const updateConfig = async (
  t: TFunction<'translation', undefined, 'translation'>,
  body: any
) => {
  if (body.state) {
    const stateName = body.state.toLowerCase()
    if (
      commonStore.settings.language !== 'zh' &&
      (stateName.includes('chn') || stateName.includes('chinese'))
    ) {
      toast(t('Note: You are using a Chinese state'), {
        type: 'warning',
        toastId: 'state_warning',
      })
    } else if (
      commonStore.settings.language !== 'dev' &&
      (stateName.includes('eng') || stateName.includes('english'))
    ) {
      toast(t('Note: You are using an English state'), {
        type: 'warning',
        toastId: 'state_warning',
      })
    } else if (
      commonStore.settings.language !== 'ja' &&
      (stateName.includes('jpn') || stateName.includes('japanese'))
    ) {
      toast(t('Note: You are using a Japanese state'), {
        type: 'warning',
        toastId: 'state_warning',
      })
    }
  }

  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort
  return fetch(`http://127.0.0.1:${port}/update-config`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
}

export const getStatus = async (
  timeout?: number
): Promise<Status | undefined> => {
  const controller = new AbortController()
  if (timeout) setTimeout(() => controller.abort(), timeout)

  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort
  let ret: Status | undefined
  await fetch(`http://127.0.0.1:${port}/status`, { signal: controller.signal })
    .then((r) => r.json())
    .then((data) => {
      ret = data
    })
    .catch(() => {})
  return ret
}
