import { FC } from 'react'
import {
  Button,
  Menu,
  MenuDivider,
  MenuGroup,
  MenuGroupHeader,
  MenuItem,
  MenuList,
  MenuPopover,
  MenuTrigger,
  PresenceBadge,
} from '@fluentui/react-components'
import {
  bundleIcon,
  Chat20Filled,
  Chat20Regular,
  ClipboardEdit20Filled,
  ClipboardEdit20Regular,
  CodeBlock20Filled,
  CodeBlock20Regular,
  MusicNote220Filled,
  MusicNote220Regular,
} from '@fluentui/react-icons'
import classNames from 'classnames'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import commonStore from '../stores/commonStore'

export const ActionSelector: FC = observer(() => {
  const { t } = useTranslation()

  return (
    <Menu>
      <MenuTrigger>
        <Button>{t('Quick Start / Select Config')}</Button>
      </MenuTrigger>
      <MenuPopover>
        <MenuList>
          <SectionQuickStart />
          <MenuDivider />
          <SectionCustomized />
        </MenuList>
      </MenuPopover>
    </Menu>
  )
})

const SectionQuickStart: FC = observer(() => {
  const { t } = useTranslation()

  const ChatIcon = bundleIcon(Chat20Filled, Chat20Regular)
  const ClipboardEditIcon = bundleIcon(
    ClipboardEdit20Filled,
    ClipboardEdit20Regular
  )
  const MusicNote220Icon = bundleIcon(MusicNote220Filled, MusicNote220Regular)
  const CodeBlock20Icon = bundleIcon(CodeBlock20Filled, CodeBlock20Regular)

  return (
    <MenuGroup>
      <MenuGroupHeader>{t('Quick Start')}</MenuGroupHeader>
      <MenuItem
        icon={<ChatIcon />}
        onClick={() => {
          // TODO: Use navigator & global function to implement this function
        }}
      >
        {t('Chat')}
      </MenuItem>
      <MenuItem
        icon={<ClipboardEditIcon />}
        onClick={() => {
          // TODO: Use navigator & global function to implement this function
        }}
      >
        {t('Completion')}
      </MenuItem>
      <MenuItem
        icon={<MusicNote220Icon />}
        onClick={() => {
          // TODO: Use navigator & global function to implement this function
        }}
      >
        {t('Composition')}
      </MenuItem>
      <MenuItem
        icon={<CodeBlock20Icon />}
        onClick={() => {
          // TODO: Use navigator & global function to implement this function
        }}
      >
        {t('Function call')}
      </MenuItem>
    </MenuGroup>
  )
})

const SectionCustomized: FC = observer(() => {
  const { t } = useTranslation()

  const selectedConfigIndex = commonStore.currentModelConfigIndex
  const selectedConfig = commonStore.modelConfigs[selectedConfigIndex]

  return (
    <MenuGroup>
      <MenuGroupHeader>{t('Customized Config')}</MenuGroupHeader>
      <div className={classNames('text-center')}>{selectedConfig.name}</div>
      <Menu>
        <MenuTrigger>
          <MenuItem>{t('All Configs')}</MenuItem>
        </MenuTrigger>
        <MenuPopover>
          <MenuList>
            {commonStore.modelConfigs.map((config, index) => {
              return (
                <MenuItem
                  key={config.name}
                  onClick={() => {
                    commonStore.setCurrentConfigIndex(index)
                  }}
                >
                  <div className="flex grow justify-between">
                    {config.name}
                    {commonStore.modelSourceList.find(
                      (item) => item.name === config.modelParameters.modelName
                    )?.isComplete && <PresenceBadge status="available" />}
                  </div>
                </MenuItem>
              )
            })}
          </MenuList>
        </MenuPopover>
      </Menu>
    </MenuGroup>
  )
})
