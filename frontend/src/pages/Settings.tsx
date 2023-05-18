import React, {FC} from 'react';
import {Page} from '../components/Page';
import {Dropdown, Option, Switch} from '@fluentui/react-components';
import {Labeled} from '../components/Labeled';
import commonStore from '../stores/commonStore';
import {observer} from 'mobx-react-lite';
import {UpdateApp} from '../../wailsjs/go/backend_golang/App';
import {useTranslation} from 'react-i18next';
import {Language, Languages} from '../utils';

export const Settings: FC = observer(() => {
  const {t, i18n} = useTranslation();

  return (
    <Page title={t('Settings')} content={
      <div className="flex flex-col gap-2 overflow-hidden">
        <Labeled label="Language" flex spaceBetween content={
          <Dropdown style={{minWidth: 0}} listbox={{style: {minWidth: 0}}}
                    value={Languages[commonStore.settings.language]}
                    selectedOptions={[Languages[commonStore.settings.language]]}
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
        }/>
        <Labeled label="Dark Mode" flex spaceBetween content={
          <Switch checked={commonStore.settings.darkMode}
                  onChange={(e, data) => {
                    commonStore.setSettings({
                      darkMode: data.checked
                    });
                  }}/>
        }/>
        <Labeled label="Automatic Updates Check" flex spaceBetween content={
          <Switch checked={commonStore.settings.autoUpdatesCheck}
                  onChange={(e, data) => {
                    commonStore.setSettings({
                      autoUpdatesCheck: data.checked
                    });
                    if (data.checked)
                      UpdateApp('http://localhost:34115/dist/RWKV-Runner.exe'); //TODO
                  }}/>
        }/>
      </div>
    }/>
  );
});
