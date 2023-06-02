import React, { FC, useEffect, useRef } from 'react';
import { Page } from '../components/Page';
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  Dropdown,
  Input,
  Option,
  Switch
} from '@fluentui/react-components';
import { Labeled } from '../components/Labeled';
import commonStore from '../stores/commonStore';
import { observer } from 'mobx-react-lite';
import { useTranslation } from 'react-i18next';
import { checkUpdate } from '../utils';

export const Languages = {
  dev: 'English', // i18n default
  zh: '简体中文'
};

export type Language = keyof typeof Languages;

export type SettingsType = {
  language: Language
  darkMode: boolean
  autoUpdatesCheck: boolean
  giteeUpdatesSource: boolean
  cnMirror: boolean
  host: string
  customModelsPath: string
  customPythonPath: string
}

export const Settings: FC = observer(() => {
  const { t, i18n } = useTranslation();
  const advancedHeaderRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (advancedHeaderRef.current)
      (advancedHeaderRef.current.firstElementChild as HTMLElement).style.padding = '0';
  }, []);

  return (
    <Page title={t('Settings')} content={
      <div className="flex flex-col gap-2 overflow-hidden">
        <Labeled label={t('Language')} flex spaceBetween content={
          <Dropdown style={{ minWidth: 0 }} listbox={{ style: { minWidth: 0 } }}
            value={Languages[commonStore.settings.language]}
            selectedOptions={[commonStore.settings.language]}
            onOptionSelect={(_, data) => {
              if (data.optionValue) {
                const lang = data.optionValue as Language;
                commonStore.setSettings({
                  language: lang
                });
                i18n.changeLanguage(lang);
              }
            }}>
            {
              Object.entries(Languages).map(([langKey, desc]) =>
                <Option key={langKey} value={langKey}>{desc}</Option>)
            }
          </Dropdown>
        } />
        <Labeled label={t('Dark Mode')} flex spaceBetween content={
          <Switch checked={commonStore.settings.darkMode}
            onChange={(e, data) => {
              commonStore.setSettings({
                darkMode: data.checked
              });
            }} />
        } />
        <Labeled label={t('Automatic Updates Check')} flex spaceBetween content={
          <Switch checked={commonStore.settings.autoUpdatesCheck}
            onChange={(e, data) => {
              commonStore.setSettings({
                autoUpdatesCheck: data.checked
              });
              if (data.checked)
                checkUpdate(true);
            }} />
        } />
        {
          commonStore.settings.language === 'zh' &&
          <Labeled label={t('Use Gitee Updates Source')} flex spaceBetween content={
            <Switch checked={commonStore.settings.giteeUpdatesSource}
              onChange={(e, data) => {
                commonStore.setSettings({
                  giteeUpdatesSource: data.checked
                });
              }} />
          } />
        }
        {
          commonStore.settings.language === 'zh' &&
          <Labeled label={t('Use Tsinghua Pip Mirrors')} flex spaceBetween content={
            <Switch checked={commonStore.settings.cnMirror}
              onChange={(e, data) => {
                commonStore.setSettings({
                  cnMirror: data.checked
                });
              }} />
          } />
        }
        <Labeled label={t('Allow external access to the API (service must be restarted)')} flex spaceBetween content={
          <Switch checked={commonStore.settings.host !== '127.0.0.1'}
            onChange={(e, data) => {
              commonStore.setSettings({
                host: data.checked ? '0.0.0.0' : '127.0.0.1'
              });
            }} />
        } />
        <Accordion collapsible>
          <AccordionItem value="1">
            <AccordionHeader ref={advancedHeaderRef} size="large">{t('Advanced')}</AccordionHeader>
            <AccordionPanel>
              <div className="flex flex-col gap-2 overflow-hidden">
                {commonStore.platform !== 'darwin' &&
                  <Labeled label={t('Custom Models Path')}
                    content={
                      <Input className="grow" placeholder="./models" value={commonStore.settings.customModelsPath}
                        onChange={(e, data) => {
                          commonStore.setSettings({
                            customModelsPath: data.value
                          });
                        }} />
                    } />
                }
                <Labeled label={t('Custom Python Path')}
                  content={
                    <Input className="grow" placeholder="./py310/python" value={commonStore.settings.customPythonPath}
                      onChange={(e, data) => {
                        commonStore.setSettings({
                          customPythonPath: data.value
                        });
                      }} />
                  } />
              </div>
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
      </div>
    } />
  );
});
