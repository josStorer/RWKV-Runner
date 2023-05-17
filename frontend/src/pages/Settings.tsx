import React, {FC} from 'react';
import {Page} from '../components/Page';
import {Dropdown, Option, Switch} from '@fluentui/react-components';
import {Labeled} from '../components/Labeled';
import commonStore from '../stores/commonStore';
import {observer} from 'mobx-react-lite';

export const Settings: FC = observer(() => {
  return (
    <Page title="Settings" content={
      <div className="flex flex-col gap-2 overflow-hidden">
        <Labeled label="Language" flex spaceBetween content={
          <Dropdown style={{minWidth: 0}} listbox={{style: {minWidth: 0}}}
                    value="English"
                    selectedOptions={['English']}
                    onOptionSelect={(_, data) => {
                      if (data.optionText) {
                      }
                    }}>
            <Option>English</Option>
            <Option>简体中文</Option>
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
                  }}/>
        }/>
      </div>
    }/>
  );
});
