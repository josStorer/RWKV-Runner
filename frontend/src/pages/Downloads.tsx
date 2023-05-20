import React, {FC} from 'react';
import {useTranslation} from 'react-i18next';
import {Page} from '../components/Page';
import {observer} from 'mobx-react-lite';
import commonStore from '../stores/commonStore';
import {Divider, Field, ProgressBar} from '@fluentui/react-components';
import {bytesToGb, bytesToMb} from '../utils';
import {ToolTipButton} from '../components/ToolTipButton';
import {Folder20Regular, Pause20Regular, Play20Regular} from '@fluentui/react-icons';
import {ContinueDownload, OpenFileFolder, PauseDownload} from '../../wailsjs/go/backend_golang/App';

export const Downloads: FC = observer(() => {
  const {t} = useTranslation();

  return (
    <Page title={t('Downloads')} content={
      <div className="flex flex-col gap-2 overflow-y-auto overflow-x-hidden p-1">
        {commonStore.downloadList.map((status, index) => (
          <div className="flex flex-col gap-1">
            <Field
              key={index}
              label={`${status.downloading ? (t('Downloading') + ': ') : ''}${status.name}`}
              validationMessage={`${status.progress.toFixed(2)}% - ${bytesToGb(status.transferred) + 'GB'}/${bytesToGb(status.size) + 'GB'} - ${status.downloading ? bytesToMb(status.speed) : 0}MB/s - ${status.url}`}
              validationState={status.done ? 'success' : 'none'}
            >
              <div className="flex items-center gap-2">
                <ProgressBar className="grow" value={status.progress} max={100}/>
                {!status.done &&
                  <ToolTipButton desc={status.downloading ? t('Pause') : t('Continue')}
                                 icon={status.downloading ? <Pause20Regular/> : <Play20Regular/>}
                                 onClick={() => {
                                   if (status.downloading)
                                     PauseDownload(status.url);
                                   else
                                     ContinueDownload(status.url);
                                 }}/>}
                <ToolTipButton desc={t('Open Folder')} icon={<Folder20Regular/>} onClick={() => {
                  OpenFileFolder(status.path);
                }}/>
              </div>
            </Field>
            <Divider style={{flexGrow: 0}}/>
          </div>
        ))
        }
      </div>
    }/>
  );
});
