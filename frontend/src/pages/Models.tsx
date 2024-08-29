import React, { FC, useEffect, useState } from 'react'
import {
  Button,
  Checkbox,
  createTableColumn,
  DataGrid,
  DataGridBody,
  DataGridCell,
  DataGridHeader,
  DataGridHeaderCell,
  DataGridRow,
  TableCellLayout,
  TableColumnDefinition,
  Text,
  Textarea,
} from '@fluentui/react-components'
import {
  ArrowClockwise20Regular,
  ArrowDownload20Regular,
  Folder20Regular,
  Open20Regular,
} from '@fluentui/react-icons'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router'
import {
  AddToDownloadList,
  OpenFileFolder,
} from '../../wailsjs/go/backend_golang/App'
import { BrowserOpenURL } from '../../wailsjs/runtime'
import { Page } from '../components/Page'
import { ToolTipButton } from '../components/ToolTipButton'
import commonStore from '../stores/commonStore'
import { ModelSourceItem } from '../types/models'
import {
  bytesToGb,
  getHfDownloadUrl,
  refreshModels,
  saveConfigs,
  toastWithButton,
} from '../utils'

const columns: TableColumnDefinition<ModelSourceItem>[] = [
  createTableColumn<ModelSourceItem>({
    columnId: 'file',
    compare: (a, b) => {
      return a.name.localeCompare(b.name)
    },
    renderHeaderCell: () => {
      const { t } = useTranslation()

      return t('File')
    },
    renderCell: (item) => {
      return (
        <TableCellLayout className="break-all">{item.name}</TableCellLayout>
      )
    },
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'desc',
    compare: (a, b) => {
      const lang: string = commonStore.settings.language

      if (a.desc && b.desc) {
        if (lang in a.desc && lang in b.desc && a.desc[lang] && b.desc[lang])
          return b.desc[lang]!.localeCompare(a.desc[lang]!)
        else if (
          'en' in a.desc &&
          'en' in b.desc &&
          a.desc['en'] &&
          b.desc['en']
        )
          return b.desc['en']!.localeCompare(a.desc['en']!)
      }
      return 0
    },
    renderHeaderCell: () => {
      const { t } = useTranslation()

      return t('Desc')
    },
    renderCell: (item) => {
      const lang: string = commonStore.settings.language

      return (
        <TableCellLayout>
          {item.desc &&
            (lang in item.desc
              ? item.desc[lang]
              : 'en' in item.desc && item.desc['en'])}
        </TableCellLayout>
      )
    },
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'size',
    compare: (a, b) => {
      return a.size - b.size
    },
    renderHeaderCell: () => {
      const { t } = useTranslation()

      return t('Size')
    },
    renderCell: (item) => {
      return <TableCellLayout>{bytesToGb(item.size) + 'GB'}</TableCellLayout>
    },
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'lastUpdated',
    compare: (a, b) => {
      if (!a.lastUpdatedMs) a.lastUpdatedMs = Date.parse(a.lastUpdated)
      if (!b.lastUpdatedMs) b.lastUpdatedMs = Date.parse(b.lastUpdated)
      return b.lastUpdatedMs - a.lastUpdatedMs
    },
    renderHeaderCell: () => {
      const { t } = useTranslation()

      return t('Last updated')
    },

    renderCell: (item) => {
      return new Date(item.lastUpdated).toLocaleString()
    },
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'actions',
    compare: (a, b) => {
      return a.isComplete ? -1 : 1
    },
    renderHeaderCell: () => {
      const { t } = useTranslation()

      return t('Actions')
    },
    renderCell: (item) => {
      const { t } = useTranslation()
      const navigate = useNavigate()

      return (
        <TableCellLayout>
          <div className="flex gap-1">
            {item.isComplete && (
              <ToolTipButton
                desc={t('Open Folder')}
                icon={<Folder20Regular />}
                onClick={() => {
                  OpenFileFolder(
                    `${commonStore.settings.customModelsPath}/${item.name}`
                  )
                }}
              />
            )}
            {item.downloadUrl && !item.isComplete && (
              <ToolTipButton
                desc={t('Download')}
                icon={<ArrowDownload20Regular />}
                onClick={() => {
                  toastWithButton(
                    `${t('Downloading')} ${item.name}`,
                    t('Check'),
                    () => {
                      navigate({ pathname: '/downloads' })
                    },
                    { autoClose: 3000 }
                  )
                  AddToDownloadList(
                    `${commonStore.settings.customModelsPath}/${item.name}`,
                    getHfDownloadUrl(item.downloadUrl!)
                  )
                }}
              />
            )}
            {item.url && (
              <ToolTipButton
                desc={t('Open Url')}
                icon={<Open20Regular />}
                onClick={() => {
                  BrowserOpenURL(item.url!)
                }}
              />
            )}
          </div>
        </TableCellLayout>
      )
    },
  }),
]

const getTags = () => {
  return Array.from(
    new Set([
      'Recommended',
      'Official',
      ...commonStore.modelSourceList
        .filter((item) => !item.hide || item.isComplete)
        .map((item) => item.tags || [])
        .flat()
        .filter((i) => !i.includes('Other') && !i.includes('Local')),
      'Other',
      'Local',
    ])
  )
}

const getCurrentModelList = () => {
  if (commonStore.activeModelListTags.length === 0)
    return commonStore.modelSourceList
  else
    return commonStore.modelSourceList.filter((item) =>
      commonStore.activeModelListTags.some((tag) => item.tags?.includes(tag))
    )
}

const Models: FC = observer(() => {
  const { t } = useTranslation()
  const [tags, setTags] = useState<Array<string>>(getTags())
  const [modelSourceList, setModelSourceList] = useState<ModelSourceItem[]>(
    getCurrentModelList()
  )

  useEffect(() => {
    setTags(getTags())
  }, [commonStore.modelSourceList])

  useEffect(() => {
    setModelSourceList(getCurrentModelList())
  }, [commonStore.modelSourceList, commonStore.activeModelListTags])

  return (
    <Page
      title={t('Models')}
      content={
        <div className="flex flex-col gap-2 overflow-hidden">
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <Text weight="medium">{t('Model Source Manifest List')}</Text>
              <div className="flex">
                {commonStore.settings.language === 'zh' && (
                  <Checkbox
                    className="select-none"
                    size="large"
                    label={t('Use Hugging Face Mirror')}
                    checked={commonStore.settings.useHfMirror}
                    onChange={(_, data) => {
                      commonStore.setSettings({
                        useHfMirror: data.checked as boolean,
                      })
                    }}
                  />
                )}
                <ToolTipButton
                  desc={t('Refresh')}
                  icon={<ArrowClockwise20Regular />}
                  onClick={() => {
                    refreshModels(false)
                    saveConfigs()
                  }}
                />
              </div>
            </div>
            <Text size={100}>
              {t(
                'Provide JSON file URLs for the models manifest. Separate URLs with semicolons. The "models" field in JSON files will be parsed into the following table.'
              )}
            </Text>
            <Textarea
              size="large"
              resize="vertical"
              value={commonStore.modelSourceManifestList}
              onChange={(e, data) =>
                commonStore.setModelSourceManifestList(data.value)
              }
            />
          </div>
          <div
            className="flex flex-wrap gap-2 overflow-y-auto"
            style={{ minHeight: '88px' }}
          >
            {tags.map((tag) => (
              <div key={tag} className="mt-auto">
                <Button
                  appearance={
                    commonStore.activeModelListTags.includes(tag)
                      ? 'primary'
                      : 'secondary'
                  }
                  onClick={() => {
                    if (commonStore.activeModelListTags.includes(tag))
                      commonStore.setActiveModelListTags(
                        commonStore.activeModelListTags.filter((t) => t !== tag)
                      )
                    else
                      commonStore.setActiveModelListTags([
                        ...commonStore.activeModelListTags,
                        tag,
                      ])
                  }}
                >
                  {t(tag)}
                </Button>
              </div>
            ))}
          </div>
          <div className="flex grow overflow-hidden">
            <DataGrid
              items={modelSourceList}
              columns={columns}
              sortable={true}
              defaultSortState={{
                sortColumn: 'actions',
                sortDirection: 'ascending',
              }}
              style={{ display: 'flex' }}
              className="w-full flex-col"
            >
              <DataGridHeader>
                <DataGridRow>
                  {({ renderHeaderCell }) => (
                    <DataGridHeaderCell>
                      {renderHeaderCell()}
                    </DataGridHeaderCell>
                  )}
                </DataGridRow>
              </DataGridHeader>
              <div className="overflow-y-auto overflow-x-hidden">
                <DataGridBody<ModelSourceItem>>
                  {({ item, rowId }) =>
                    (!item.hide || item.isComplete) && (
                      <DataGridRow<ModelSourceItem> key={rowId}>
                        {({ renderCell }) => (
                          <DataGridCell>{renderCell(item)}</DataGridCell>
                        )}
                      </DataGridRow>
                    )
                  }
                </DataGridBody>
              </div>
            </DataGrid>
          </div>
        </div>
      }
    />
  )
})

export default Models
