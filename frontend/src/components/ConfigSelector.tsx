import {FC} from 'react';
import {observer} from 'mobx-react-lite';
import {Dropdown, Option} from '@fluentui/react-components';
import commonStore from '../stores/commonStore';

export const ConfigSelector: FC<{ size?: 'small' | 'medium' | 'large' }> = observer(({size}) => {
  return <Dropdown size={size} style={{minWidth: 0}} listbox={{style: {minWidth: 0}}}
                   value={commonStore.getCurrentModelConfig().name}
                   selectedOptions={[commonStore.currentModelConfigIndex.toString()]}
                   onOptionSelect={(_, data) => {
                     if (data.optionValue)
                       commonStore.setCurrentConfigIndex(Number(data.optionValue));
                   }}>
    {commonStore.modelConfigs.map((config, index) =>
      <Option key={index} value={index.toString()}>{config.name}</Option>
    )}
  </Dropdown>;
});