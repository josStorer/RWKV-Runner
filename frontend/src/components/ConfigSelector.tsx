import { FC } from 'react'
import { Dropdown, Option, PresenceBadge } from '@fluentui/react-components'
import { observer } from 'mobx-react-lite'
import commonStore from '../stores/commonStore'

export const ConfigSelector: FC<{ size?: 'small' | 'medium' | 'large' }> =
  observer(({ size }) => {
    return (
      <Dropdown
        size={size}
        style={{ minWidth: 0 }}
        listbox={{ style: { minWidth: 'fit-content' } }}
        value={commonStore.getCurrentModelConfig().name}
        selectedOptions={[commonStore.currentModelConfigIndex.toString()]}
        onOptionSelect={(_, data) => {
          if (data.optionValue)
            commonStore.setCurrentConfigIndex(Number(data.optionValue))
        }}
      >
        {commonStore.modelConfigs.map((config, index) => (
          <Option key={index} value={index.toString()} text={config.name}>
            <div className="flex grow justify-between">
              {config.name}
              {commonStore.modelSourceList.find(
                (item) => item.name === config.modelParameters.modelName
              )?.isComplete && <PresenceBadge status="available" />}
            </div>
          </Option>
        ))}
      </Dropdown>
    )
  })
