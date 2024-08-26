import React, { FC, useEffect } from 'react'
import { Divider, Field, ProgressBar } from '@fluentui/react-components'
import {
  Folder20Regular,
  Pause20Regular,
  Play20Regular,
} from '@fluentui/react-icons'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import {
  AddToDownloadList,
  OpenFileFolder,
  PauseDownload,
} from '../../wailsjs/go/backend_golang/App'
import { Page } from '../components/Page'
import { ToolTipButton } from '../components/ToolTipButton'
import commonStore from '../stores/commonStore'
import { bytesToGb, bytesToKb, bytesToMb, refreshLocalModels } from '../utils'

const Downloads: FC = observer(() => {
  const { t } = useTranslation()
  const finishedModelsLen = commonStore.downloadList.filter(
    (status) => status.done && status.name.endsWith('.pth')
  ).length
  useEffect(() => {
    if (finishedModelsLen > 0)
      refreshLocalModels({ models: commonStore.modelSourceList }, false)
    console.log('finishedModelsLen:', finishedModelsLen)
  }, [finishedModelsLen])

  let displayList = commonStore.downloadList.slice()
  const downloadListNames = displayList.map((s) => s.name)
  commonStore.lastUnfinishedModelDownloads.forEach((status) => {
    const unfinishedIndex = downloadListNames.indexOf(status.name)
    if (unfinishedIndex === -1) {
      displayList.push(status)
    } else {
      const unfinishedStatus = displayList[unfinishedIndex]
      if (unfinishedStatus.transferred < status.transferred) {
        status.downloading = unfinishedStatus.downloading
        delete displayList[unfinishedIndex]
        displayList.push(status)
      }
    }
  })
  displayList = displayList.reverse()

  return (
    <Page
      title={t('Downloads')}
      content={
        <div className="flex flex-col gap-2 overflow-y-auto overflow-x-hidden p-1">
          {displayList.map((status, index) => {
            const downloadProgress = `${status.progress.toFixed(2)}%`
            const downloadSpeed = `${
              status.downloading ? bytesToMb(status.speed) : '0'
            }MB/s`
            let downloadDetails: string
            if (status.size < 1024 * 1024)
              downloadDetails = `${bytesToKb(status.transferred) + 'KB'}/${
                bytesToKb(status.size) + 'KB'
              }`
            else if (status.size < 1024 * 1024 * 1024)
              downloadDetails = `${bytesToMb(status.transferred) + 'MB'}/${
                bytesToMb(status.size) + 'MB'
              }`
            else
              downloadDetails = `${bytesToGb(status.transferred) + 'GB'}/${
                bytesToGb(status.size) + 'GB'
              }`

            return (
              <div className="flex flex-col gap-1" key={index}>
                <Field
                  label={`${status.downloading ? t('Downloading') + ': ' : ''}${
                    status.name
                  }`}
                  validationMessage={`${downloadProgress} - ${downloadDetails} - ${downloadSpeed} - ${status.url}`}
                  validationState={status.done ? 'success' : 'none'}
                >
                  <div className="flex items-center gap-2">
                    <ProgressBar
                      className="grow"
                      value={status.progress}
                      max={100}
                    />
                    {!status.done && (
                      <ToolTipButton
                        desc={status.downloading ? t('Pause') : t('Resume')}
                        icon={
                          status.downloading ? (
                            <Pause20Regular />
                          ) : (
                            <Play20Regular />
                          )
                        }
                        onClick={() => {
                          if (status.downloading) PauseDownload(status.url)
                          else AddToDownloadList(status.path, status.url)
                        }}
                      />
                    )}
                    <ToolTipButton
                      desc={t('Open Folder')}
                      icon={<Folder20Regular />}
                      onClick={() => {
                        OpenFileFolder(status.path)
                      }}
                    />
                  </div>
                </Field>
                <Divider style={{ flexGrow: 0 }} />
              </div>
            )
          })}
        </div>
      }
    />
  )
})

export default Downloads
