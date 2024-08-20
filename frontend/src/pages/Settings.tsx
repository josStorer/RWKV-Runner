import React, { FC, useEffect, useRef } from 'react'
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Dropdown,
  Input,
  Option,
  Switch,
} from '@fluentui/react-components'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { toast } from 'react-toastify'
import { RestartApp } from '../../wailsjs/go/backend_golang/App'
import { Labeled } from '../components/Labeled'
import { Page } from '../components/Page'
import commonStore from '../stores/commonStore'
import { Language, Languages } from '../types/settings'
import { checkUpdate, toastWithButton } from '../utils'

export const GeneralSettings: FC = observer(() => {
  const { t } = useTranslation()

  return (
    <div className="flex flex-col gap-2">
      <Labeled
        label={t('Language')}
        flex
        spaceBetween
        content={
          <Dropdown
            style={{ minWidth: 0 }}
            listbox={{ style: { minWidth: 'fit-content' } }}
            value={Languages[commonStore.settings.language]}
            selectedOptions={[commonStore.settings.language]}
            onOptionSelect={(_, data) => {
              if (data.optionValue) {
                const lang = data.optionValue as Language
                commonStore.setSettings({
                  language: lang,
                })
              }
            }}
          >
            {Object.entries(Languages).map(([langKey, desc]) => (
              <Option key={langKey} value={langKey}>
                {desc}
              </Option>
            ))}
          </Dropdown>
        }
      />
      {commonStore.platform === 'windows' && (
        <Labeled
          label={t('DPI Scaling')}
          flex
          spaceBetween
          content={
            <Dropdown
              style={{ minWidth: 0 }}
              listbox={{ style: { minWidth: 'fit-content' } }}
              value={commonStore.settings.dpiScaling + '%'}
              selectedOptions={[commonStore.settings.dpiScaling.toString()]}
              onOptionSelect={(_, data) => {
                if (data.optionValue) {
                  commonStore.setSettings({
                    dpiScaling: Number(data.optionValue),
                  })
                  toastWithButton(
                    t('Restart the app to apply DPI Scaling.'),
                    t('Restart'),
                    () => {
                      RestartApp()
                    },
                    {
                      autoClose: 5000,
                    }
                  )
                }
              }}
            >
              {Array.from({ length: 7 }, (_, i) => (i + 2) * 25).map((v, i) => (
                <Option key={i} value={v.toString()}>
                  {v + '%'}
                </Option>
              ))}
            </Dropdown>
          }
        />
      )}
      <Labeled
        label={t('Dark Mode')}
        flex
        spaceBetween
        content={
          <Switch
            checked={commonStore.settings.darkMode}
            onChange={(e, data) => {
              commonStore.setSettings({
                darkMode: data.checked,
              })
            }}
          />
        }
      />
    </div>
  )
})

export const AdvancedGeneralSettings: FC = observer(() => {
  const { t } = useTranslation()

  return (
    <div className="flex flex-col gap-2">
      <Labeled
        label={'API URL'}
        content={
          <div className="flex gap-2">
            <Input
              style={{ minWidth: 0 }}
              className="grow"
              value={commonStore.settings.apiUrl}
              onChange={(e, data) => {
                commonStore.setSettings({
                  apiUrl: data.value,
                })
              }}
            />
            <Dropdown
              style={{ minWidth: '33px' }}
              listbox={{ style: { minWidth: 'fit-content' } }}
              value="..."
              selectedOptions={[]}
              expandIcon={null}
              onOptionSelect={(_, data) => {
                commonStore.setSettings({
                  apiUrl: data.optionValue,
                })
                if (data.optionText === 'OpenAI') {
                  if (commonStore.settings.apiChatModelName === 'rwkv')
                    commonStore.setSettings({
                      apiChatModelName: 'gpt-3.5-turbo',
                    })
                  if (commonStore.settings.apiCompletionModelName === 'rwkv')
                    commonStore.setSettings({
                      apiCompletionModelName: 'gpt-3.5-turbo-instruct',
                    })
                } else if (data.optionText === 'RWKV') {
                  if (commonStore.settings.apiChatModelName === 'gpt-3.5-turbo')
                    commonStore.setSettings({
                      apiChatModelName: 'rwkv',
                    })
                  if (
                    commonStore.settings.apiCompletionModelName ===
                      'gpt-3.5-turbo-instruct' ||
                    commonStore.settings.apiCompletionModelName ===
                      'text-davinci-003'
                  )
                    commonStore.setSettings({
                      apiCompletionModelName: 'rwkv',
                    })
                } else if (data.optionText === 'Ollama') {
                  toast(
                    t(
                      "Don't forget to correctly fill in your Ollama API Chat Model Name."
                    ),
                    { type: 'info' }
                  )
                }
              }}
            >
              <Option value="">{t('Localhost')!}</Option>
              <Option value="https://rwkv.ai-creator.net/chntuned">RWKV</Option>
              <Option value="https://api.openai.com">OpenAI</Option>
              <Option value="http://localhost:11434">Ollama</Option>
            </Dropdown>
          </div>
        }
      />
      <Labeled
        label={'API Key'}
        content={
          <Input
            type="password"
            className="grow"
            placeholder="sk-"
            value={commonStore.settings.apiKey}
            onChange={(e, data) => {
              commonStore.setSettings({
                apiKey: data.value,
              })
            }}
          />
        }
      />
      <Labeled
        label={t('API Chat Model Name')}
        content={
          <div className="flex gap-2">
            <Input
              style={{ minWidth: 0 }}
              className="grow"
              placeholder="rwkv"
              value={commonStore.settings.apiChatModelName}
              onChange={(e, data) => {
                commonStore.setSettings({
                  apiChatModelName: data.value,
                })
              }}
            />
            <Dropdown
              style={{ minWidth: '33px' }}
              listbox={{ style: { minWidth: 'fit-content' } }}
              value="..."
              selectedOptions={[]}
              expandIcon={null}
              onOptionSelect={(_, data) => {
                if (data.optionValue) {
                  commonStore.setSettings({
                    apiChatModelName: data.optionValue,
                  })
                }
              }}
            >
              {[
                'rwkv',
                'gpt-4-1106-preview',
                'gpt-4',
                'gpt-4-32k',
                'gpt-4-0613',
                'gpt-4-32k-0613',
                'gpt-3.5-turbo-1106',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-16k',
              ].map((v, i) => (
                <Option key={i} value={v}>
                  {v}
                </Option>
              ))}
            </Dropdown>
          </div>
        }
      />
      <Labeled
        label={t('API Completion Model Name')}
        content={
          <div className="flex gap-2">
            <Input
              style={{ minWidth: 0 }}
              className="grow"
              placeholder="rwkv"
              value={commonStore.settings.apiCompletionModelName}
              onChange={(e, data) => {
                commonStore.setSettings({
                  apiCompletionModelName: data.value,
                })
              }}
            />
            <Dropdown
              style={{ minWidth: '33px' }}
              listbox={{ style: { minWidth: 'fit-content', minHeight: 0 } }}
              value="..."
              selectedOptions={[]}
              expandIcon={null}
              onOptionSelect={(_, data) => {
                if (data.optionValue) {
                  commonStore.setSettings({
                    apiCompletionModelName: data.optionValue,
                  })
                }
              }}
            >
              {[
                'rwkv',
                'gpt-3.5-turbo-instruct',
                'text-davinci-003',
                'text-davinci-002',
                'code-davinci-002',
                'text-curie-001',
                'text-babbage-001',
                'text-ada-001',
              ].map((v, i) => (
                <Option key={i} value={v}>
                  {v}
                </Option>
              ))}
            </Dropdown>
          </div>
        }
      />
      <Labeled
        label={t('Core API URL')}
        desc={t(
          "Override core API URL(/chat/completions and /completions). If you don't know what this is, leave it blank."
        )}
        content={
          <Input
            style={{ minWidth: 0 }}
            className="grow"
            value={commonStore.settings.coreApiUrl}
            onChange={(e, data) => {
              commonStore.setSettings({
                coreApiUrl: data.value,
              })
            }}
          />
        }
      />
    </div>
  )
})

const Settings: FC = observer(() => {
  const { t } = useTranslation()
  const advancedHeaderRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (advancedHeaderRef.current)
      (
        advancedHeaderRef.current.firstElementChild as HTMLElement
      ).style.padding = '0'
  }, [])

  return (
    <Page
      title={t('Settings')}
      content={
        <div className="flex flex-col gap-2 overflow-y-auto overflow-x-hidden p-1">
          {commonStore.platform === 'web' ? (
            <div className="flex flex-col gap-2">
              <GeneralSettings />
              <AdvancedGeneralSettings />
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <GeneralSettings />
              <Labeled
                label={t('Automatic Updates Check')}
                flex
                spaceBetween
                content={
                  <Switch
                    checked={commonStore.settings.autoUpdatesCheck}
                    onChange={(e, data) => {
                      commonStore.setSettings({
                        autoUpdatesCheck: data.checked,
                      })
                      if (data.checked) checkUpdate(true)
                    }}
                  />
                }
              />
              {commonStore.settings.language === 'zh' && (
                <Labeled
                  label={t('Use Gitee Updates Source')}
                  flex
                  spaceBetween
                  content={
                    <Switch
                      checked={commonStore.settings.giteeUpdatesSource}
                      onChange={(e, data) => {
                        commonStore.setSettings({
                          giteeUpdatesSource: data.checked,
                        })
                      }}
                    />
                  }
                />
              )}
              {commonStore.settings.language === 'zh' &&
                commonStore.platform !== 'linux' && (
                  <Labeled
                    label={t('Use Alibaba Cloud Pip Mirrors')}
                    flex
                    spaceBetween
                    content={
                      <Switch
                        checked={commonStore.settings.cnMirror}
                        onChange={(e, data) => {
                          commonStore.setSettings({
                            cnMirror: data.checked,
                          })
                        }}
                      />
                    }
                  />
                )}
              <Labeled
                label={t(
                  'Allow external access to the API (service must be restarted)'
                )}
                flex
                spaceBetween
                content={
                  <Switch
                    checked={commonStore.settings.host !== '127.0.0.1'}
                    onChange={(e, data) => {
                      commonStore.setSettings({
                        host: data.checked ? '0.0.0.0' : '127.0.0.1',
                      })
                    }}
                  />
                }
              />
              <Accordion
                collapsible
                openItems={!commonStore.advancedCollapsed && 'advanced'}
                onToggle={(e, data) => {
                  if (data.value === 'advanced')
                    commonStore.setAdvancedCollapsed(
                      !commonStore.advancedCollapsed
                    )
                }}
              >
                <AccordionItem value="advanced">
                  <AccordionHeader ref={advancedHeaderRef} size="large">
                    {t('Advanced')}
                  </AccordionHeader>
                  <AccordionPanel>
                    <div className="flex flex-col gap-2 overflow-hidden">
                      <Labeled
                        label={t('Custom Models Path')}
                        content={
                          <Input
                            className="grow"
                            placeholder="./models"
                            value={commonStore.settings.customModelsPath}
                            onChange={(e, data) => {
                              commonStore.setSettings({
                                customModelsPath: data.value,
                              })
                            }}
                          />
                        }
                      />
                      <Labeled
                        label={t('Custom Python Path')} // if set, will not use precompiled cuda kernel
                        content={
                          <Input
                            className="grow"
                            placeholder="./py310/python"
                            value={commonStore.settings.customPythonPath}
                            onChange={(e, data) => {
                              commonStore.setDepComplete(false)
                              commonStore.setSettings({
                                customPythonPath: data.value,
                              })
                            }}
                          />
                        }
                      />
                      <AdvancedGeneralSettings />
                    </div>
                  </AccordionPanel>
                </AccordionItem>
              </Accordion>
            </div>
          )}
        </div>
      }
    />
  )
})

export default Settings
