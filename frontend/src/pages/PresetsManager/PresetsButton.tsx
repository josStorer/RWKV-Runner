// TODO refactor

import React, {
  FC,
  lazy,
  PropsWithChildren,
  ReactElement,
  useState,
} from 'react'
import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTrigger,
  Input,
  Switch,
  Tab,
  TabList,
  Text,
} from '@fluentui/react-components'
import {
  Accessibility28Regular,
  Chat20Regular,
  ClipboardEdit20Regular,
  Delete20Regular,
  Dismiss20Regular,
  Edit20Regular,
  Globe20Regular,
} from '@fluentui/react-icons'
import { SelectTabEventHandler } from '@fluentui/react-tabs'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { toast } from 'react-toastify'
import { ClipboardGetText, ClipboardSetText } from '../../../wailsjs/runtime'
import logo from '../../assets/images/logo.png'
import { CustomToastContainer } from '../../components/CustomToastContainer'
import { Labeled } from '../../components/Labeled'
import { LazyImportComponent } from '../../components/LazyImportComponent'
import { ToolTipButton } from '../../components/ToolTipButton'
import commonStore from '../../stores/commonStore'
import { Preset, PresetsNavigationItem } from '../../types/presets'
import { absPathAsset, setActivePreset } from '../../utils'

const defaultPreset: Preset = {
  name: 'RWKV',
  tag: 'default',
  sourceUrl: '',
  desc: '',
  avatarImg: logo,
  type: 'chat',
  welcomeMessage: '',
  displayPresetMessages: true,
  messages: [],
  prompt: '',
  stop: '',
  injectStart: '',
  injectEnd: '',
  presystem: false,
  userName: '',
  assistantName: '',
}

const MessagesEditor = lazy(() => import('./MessagesEditor'))

const PresetCardFrame: FC<
  PropsWithChildren & {
    onClick?: React.MouseEventHandler<HTMLButtonElement>
    highlight?: boolean
  }
> = (props) => {
  return (
    <Button
      className="flex h-56 w-32 flex-col gap-1 break-all"
      style={{
        minWidth: 0,
        borderRadius: '0.75rem',
        justifyContent: 'unset',
        ...(props.highlight
          ? { borderColor: '#115ea3', borderWidth: '2px' }
          : {}),
      }}
      onClick={props.onClick}
    >
      {props.children}
    </Button>
  )
}

const PresetCard: FC<{
  editable: boolean
  preset: Preset
  presetIndex: number
}> = observer(({ editable, preset, presetIndex }) => {
  const { t } = useTranslation()

  return (
    <PresetCardFrame
      highlight={commonStore.activePresetIndex === presetIndex}
      onClick={(e) => {
        if (e.currentTarget.contains(e.target as Node))
          setActivePreset(
            presetIndex === -1 ? defaultPreset : preset,
            presetIndex
          )
      }}
    >
      <img
        src={absPathAsset(preset.avatarImg)}
        className="ml-auto mr-auto h-28 select-none rounded-xl"
      />
      <Text size={400}>{preset.name}</Text>
      <Text
        size={200}
        style={{
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
        }}
      >
        {preset.desc}
      </Text>
      <div className="grow" />
      <div className="flex w-full items-end justify-between">
        <div className="text-xs font-thin text-gray-500">{t(preset.tag)}</div>
        {editable ? (
          <ChatPresetEditor
            presetIndex={presetIndex}
            triggerButton={
              <ToolTipButton
                size="small"
                appearance="transparent"
                desc={t('Edit')}
                icon={<Edit20Regular />}
                onClick={(e) => {
                  e.stopPropagation()
                  commonStore.setEditingPreset({
                    ...commonStore.presets[presetIndex],
                  })
                }}
              />
            }
          />
        ) : (
          <div />
        )}
      </div>
    </PresetCardFrame>
  )
})

const ChatPresetEditor: FC<{
  triggerButton: ReactElement
  presetIndex: number
}> = observer(({ triggerButton, presetIndex }) => {
  const { t } = useTranslation()
  const [open, setOpen] = React.useState(false)
  const [showExitConfirm, setShowExitConfirm] = React.useState(false)
  const [editingMessages, setEditingMessages] = useState(false)

  if (open && !commonStore.editingPreset)
    commonStore.setEditingPreset({ ...defaultPreset })
  const editingPreset = commonStore.editingPreset!

  const setEditingPreset = (newParams: Partial<Preset>) => {
    commonStore.setEditingPreset({
      ...editingPreset,
      ...newParams,
    })
  }

  const importPreset = () => {
    ClipboardGetText()
      .then((text) => {
        try {
          if (!text.trim().startsWith('{'))
            text = new TextDecoder().decode(
              new Uint8Array(
                atob(text)
                  .split('')
                  .map((c) => c.charCodeAt(0))
              )
            )
          const preset = JSON.parse(text)
          setEditingPreset(preset)
          setEditingMessages(false)
          toast(t('Imported successfully'), {
            type: 'success',
            autoClose: 1000,
          })
        } catch (e) {
          toast(t('Failed to import. Please copy a preset to the clipboard.'), {
            type: 'error',
            autoClose: 2500,
          })
        }
      })
      .catch(() => {
        toast(t('Clipboard is empty.'), {
          type: 'info',
          autoClose: 1000,
        })
      })
  }

  const copyPreset = () => {
    ClipboardSetText(JSON.stringify(editingPreset)).then((success) => {
      if (success)
        toast(t('Successfully copied to clipboard.'), {
          type: 'success',
          autoClose: 1000,
        })
    })
  }

  const savePreset = () => {
    setOpen(false)
    setShowExitConfirm(false)
    if (presetIndex === -1) {
      commonStore.setPresets([...commonStore.presets, { ...editingPreset }])
    } else {
      commonStore.presets[presetIndex] = editingPreset
      commonStore.setPresets(commonStore.presets)
    }
    commonStore.setEditingPreset(null)
  }

  const activatePreset = () => {
    savePreset()
    setActivePreset(
      editingPreset,
      presetIndex === -1 ? commonStore.presets.length - 1 : presetIndex
    )
  }

  const deletePreset = () => {
    if (commonStore.activePresetIndex === presetIndex) {
      setActivePreset(defaultPreset, -1)
    }
    commonStore.presets.splice(presetIndex, 1)
    commonStore.setPresets(commonStore.presets)
    setOpen(false)
    setShowExitConfirm(false)
    commonStore.setEditingPreset(null)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(e, data) => {
        if (data.open) {
          setOpen(true)
        } else if (
          !commonStore.editingPreset ||
          (presetIndex === -1 &&
            JSON.stringify(editingPreset) === JSON.stringify(defaultPreset)) ||
          (presetIndex !== -1 &&
            JSON.stringify(editingPreset) ===
              JSON.stringify(commonStore.presets[presetIndex]))
        ) {
          setOpen(false)
          setShowExitConfirm(false)
          commonStore.setEditingPreset(null)
        } else {
          setShowExitConfirm(true)
        }
      }}
    >
      <DialogTrigger disableButtonEnhancement>{triggerButton}</DialogTrigger>
      <DialogSurface
        style={{
          paddingTop: 0,
          maxWidth: '80vw',
          maxHeight: '80vh',
          width: '500px',
          height: '100%',
          transform: 'unset', // override the style for the new version of @fluentui/react-components to avoid conflicts with react-beautiful-dnd
        }}
      >
        <DialogBody style={{ height: '100%', overflow: 'hidden' }}>
          {editingPreset && (
            <DialogContent className="flex flex-col gap-1 overflow-hidden">
              <CustomToastContainer />
              <Dialog open={showExitConfirm}>
                <DialogSurface style={{ transform: 'unset' }}>
                  <DialogBody>
                    <DialogContent>
                      {t(
                        'Content has been changed, are you sure you want to exit without saving?'
                      )}
                    </DialogContent>
                    <DialogActions>
                      <Button
                        appearance="secondary"
                        onClick={() => {
                          setShowExitConfirm(false)
                        }}
                      >
                        {t('Cancel')}
                      </Button>
                      <Button
                        appearance="primary"
                        onClick={() => {
                          setOpen(false)
                          setShowExitConfirm(false)
                          commonStore.setEditingPreset(null)
                        }}
                      >
                        {t('Exit without saving')}
                      </Button>
                    </DialogActions>
                  </DialogBody>
                </DialogSurface>
              </Dialog>
              <div className="flex justify-between">
                {presetIndex === -1 ? (
                  <div />
                ) : (
                  <Button
                    appearance="subtle"
                    icon={<Delete20Regular />}
                    onClick={deletePreset}
                  />
                )}
                <DialogTrigger disableButtonEnhancement>
                  <Button appearance="subtle" icon={<Dismiss20Regular />} />
                </DialogTrigger>
              </div>
              <img
                src={absPathAsset(editingPreset.avatarImg)}
                className="ml-auto mr-auto h-28 select-none rounded-xl"
              />
              <Labeled
                flex
                breakline
                label={t('Name')}
                content={
                  <div className="flex gap-2">
                    <Input
                      className="grow"
                      value={editingPreset.name}
                      onChange={(e, data) => {
                        setEditingPreset({
                          name: data.value,
                        })
                      }}
                    />
                    <Button
                      onClick={() => {
                        setEditingMessages(!editingMessages)
                      }}
                    >
                      {!editingMessages
                        ? t('Edit Character Settings')
                        : t('Go Back')}
                    </Button>
                  </div>
                }
              />
              {editingMessages ? (
                <div className="flex flex-col gap-1">
                  <Labeled
                    flex
                    spaceBetween
                    label={t('Insert default system prompt at the beginning')}
                    desc={t(
                      "Inside the model, there is a default prompt to improve the model's handling of common issues, but it may degrade the role-playing effect. You can disable this option to achieve a better role-playing effect."
                    )}
                    content={
                      <Switch
                        checked={
                          editingPreset.presystem === undefined
                            ? false
                            : editingPreset.presystem
                        }
                        onChange={(e, data) => {
                          setEditingPreset({
                            presystem: data.checked,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    flex
                    breakline
                    label={t('User Name')}
                    desc={t(
                      'The name used internally by the model when processing user message, changing this value helps improve the role-playing effect.'
                    )}
                    content={
                      <Input
                        placeholder="User"
                        value={editingPreset.userName}
                        onChange={(e, data) => {
                          setEditingPreset({
                            userName: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    flex
                    breakline
                    label={t('Assistant Name')}
                    desc={t(
                      'The name used internally by the model when processing AI message, changing this value helps improve the role-playing effect.'
                    )}
                    content={
                      <Input
                        placeholder="Assistant"
                        value={editingPreset.assistantName}
                        onChange={(e, data) => {
                          setEditingPreset({
                            assistantName: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <LazyImportComponent lazyChildren={MessagesEditor} />
                </div>
              ) : (
                <div className="flex flex-col gap-1 overflow-y-auto overflow-x-hidden p-2">
                  <Labeled
                    flex
                    breakline
                    label={`${t('Description')} (${t('Preview Only')})`}
                    content={
                      <Input
                        value={editingPreset.desc}
                        onChange={(e, data) => {
                          setEditingPreset({
                            desc: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    flex
                    breakline
                    label={t('Assistant Avatar Url')}
                    content={
                      <Input
                        value={editingPreset.avatarImg}
                        onChange={(e, data) => {
                          setEditingPreset({
                            avatarImg: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    flex
                    breakline
                    label={t('User Avatar Url')}
                    content={
                      <Input
                        value={editingPreset.userAvatarImg}
                        onChange={(e, data) => {
                          setEditingPreset({
                            userAvatarImg: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    flex
                    breakline
                    label={t('Welcome Message')}
                    content={
                      <Input
                        disabled
                        value={editingPreset.welcomeMessage}
                        onChange={(e, data) => {
                          setEditingPreset({
                            welcomeMessage: data.value,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    flex
                    spaceBetween
                    label={t('Display Preset Messages')}
                    content={
                      <Switch
                        disabled
                        checked={editingPreset.displayPresetMessages}
                        onChange={(e, data) => {
                          setEditingPreset({
                            displayPresetMessages: data.checked,
                          })
                        }}
                      />
                    }
                  />
                  <Labeled
                    flex
                    breakline
                    label={t('Tag')}
                    content={
                      <Input
                        value={editingPreset.tag}
                        onChange={(e, data) => {
                          setEditingPreset({
                            tag: data.value,
                          })
                        }}
                      />
                    }
                  />
                </div>
              )}
              <div className="grow" />
              <div className="flex justify-between">
                <Button onClick={importPreset}>{t('Import')}</Button>
                <Button onClick={copyPreset}>{t('Copy')}</Button>
              </div>
              <div className="flex justify-between">
                <Button appearance="primary" onClick={savePreset}>
                  {t('Save')}
                </Button>
                <Button appearance="primary" onClick={activatePreset}>
                  {t('Activate')}
                </Button>
              </div>
            </DialogContent>
          )}
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
})

const ChatPresets: FC = observer(() => {
  const { t } = useTranslation()

  return (
    <div className="flex flex-wrap gap-2">
      <ChatPresetEditor
        presetIndex={-1}
        triggerButton={
          <PresetCardFrame>
            <div className="flex h-full items-center">{t('New Preset')}</div>
          </PresetCardFrame>
        }
      />
      {/*TODO <PresetCardFrame>*/}
      {/*  <div className="h-full flex items-center">*/}
      {/*    {t('Import')}*/}
      {/*  </div>*/}
      {/*</PresetCardFrame>*/}
      <PresetCard preset={defaultPreset} presetIndex={-1} editable={false} />
      {commonStore.presets.map((preset, index) => {
        return (
          <PresetCard
            key={index}
            preset={preset}
            presetIndex={index}
            editable={true}
          />
        )
      })}
    </div>
  )
})

const pages: {
  [label: string]: PresetsNavigationItem
} = {
  Chat: {
    icon: <Chat20Regular />,
    element: <ChatPresets />,
  },
  Completion: {
    icon: <ClipboardEdit20Regular />,
    element: <div>In Development</div>,
  },
  Online: {
    icon: <Globe20Regular />,
    element: <div>In Development</div>,
  },
}

const PresetsManager: FC<{
  initTab: string
}> = ({ initTab }) => {
  const { t } = useTranslation()
  const [tab, setTab] = useState(initTab)

  const selectTab: SelectTabEventHandler = (e, data) =>
    typeof data.value === 'string' ? setTab(data.value) : null

  return (
    <div className="flex h-full w-full flex-col gap-2">
      <div className="flex justify-between">
        <TabList
          size="small"
          appearance="subtle"
          selectedValue={tab}
          onTabSelect={selectTab}
        >
          {Object.entries(pages).map(([label, { icon }]) => (
            <Tab icon={icon} key={label} value={label}>
              {t(label)}
            </Tab>
          ))}
        </TabList>
        <DialogTrigger disableButtonEnhancement>
          <Button appearance="subtle" icon={<Dismiss20Regular />} />
        </DialogTrigger>
      </div>
      <div className="grow overflow-y-auto overflow-x-hidden">
        {pages[tab].element}
      </div>
    </div>
  )
}

export const PresetsButton: FC<{
  tab: string
  size?: 'small' | 'medium' | 'large'
  shape?: 'rounded' | 'circular' | 'square'
  appearance?: 'secondary' | 'primary' | 'outline' | 'subtle' | 'transparent'
}> = ({ tab, size, shape, appearance }) => {
  const { t } = useTranslation()

  return (
    <Dialog>
      <DialogTrigger disableButtonEnhancement>
        <ToolTipButton
          desc={t('Presets')}
          size={size}
          shape={shape}
          appearance={appearance}
          icon={<Accessibility28Regular />}
        />
      </DialogTrigger>
      <DialogSurface
        style={{
          paddingTop: 0,
          maxWidth: '90vw',
          width: 'fit-content',
          transform: 'unset',
        }}
      >
        <DialogBody>
          <DialogContent>
            <CustomToastContainer />
            <PresetsManager initTab={tab} />
          </DialogContent>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  )
}
