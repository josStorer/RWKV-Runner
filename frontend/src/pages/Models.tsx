import React, { FC } from 'react';
import {
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
  Textarea
} from '@fluentui/react-components';
import { ToolTipButton } from '../components/ToolTipButton';
import { ArrowClockwise20Regular, ArrowDownload20Regular, Folder20Regular, Open20Regular } from '@fluentui/react-icons';
import { observer } from 'mobx-react-lite';
import commonStore from '../stores/commonStore';
import { BrowserOpenURL } from '../../wailsjs/runtime';
import { AddToDownloadList, OpenFileFolder } from '../../wailsjs/go/backend_golang/App';
import { Page } from '../components/Page';
import { bytesToGb, refreshModels, saveConfigs, toastWithButton } from '../utils';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router';

export type ModelSourceItem = {
  name: string;
  size: number;
  lastUpdated: string;
  desc?: { [lang: string]: string | undefined; };
  SHA256?: string;
  url?: string;
  downloadUrl?: string;
  isComplete?: boolean;
  isLocal?: boolean;
  localSize?: number;
  lastUpdatedMs?: number;
  hide?: boolean;
};

const columns: TableColumnDefinition<ModelSourceItem>[] = [
  createTableColumn<ModelSourceItem>({
    columnId: 'file',
    compare: (a, b) => {
      return a.name.localeCompare(b.name);
    },
    renderHeaderCell: () => {
      const { t } = useTranslation();

      return t('File');
    },
    renderCell: (item) => {
      return (
        <TableCellLayout className="break-all">
          {item.name}
        </TableCellLayout>
      );
    }
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'desc',
    compare: (a, b) => {
      const lang: string = commonStore.settings.language;

      if (a.desc && b.desc) {
        if (lang in a.desc && lang in b.desc && a.desc[lang] && b.desc[lang])
          return b.desc[lang]!.localeCompare(a.desc[lang]!);
        else if ('en' in a.desc && 'en' in b.desc && a.desc['en'] && b.desc['en'])
          return b.desc['en']!.localeCompare(a.desc['en']!);
      }
      return 0;
    },
    renderHeaderCell: () => {
      const { t } = useTranslation();

      return t('Desc');
    },
    renderCell: (item) => {
      const lang: string = commonStore.settings.language;

      return (
        <TableCellLayout>
          {item.desc &&
            (lang in item.desc ? item.desc[lang] :
              ('en' in item.desc && item.desc['en']))}
        </TableCellLayout>
      );
    }
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'size',
    compare: (a, b) => {
      return a.size - b.size;
    },
    renderHeaderCell: () => {
      const { t } = useTranslation();

      return t('Size');
    },
    renderCell: (item) => {
      return (
        <TableCellLayout>
          {bytesToGb(item.size) + 'GB'}
        </TableCellLayout>
      );
    }
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'lastUpdated',
    compare: (a, b) => {
      if (!a.lastUpdatedMs)
        a.lastUpdatedMs = Date.parse(a.lastUpdated);
      if (!b.lastUpdatedMs)
        b.lastUpdatedMs = Date.parse(b.lastUpdated);
      return b.lastUpdatedMs - a.lastUpdatedMs;
    },
    renderHeaderCell: () => {
      const { t } = useTranslation();

      return t('Last updated');
    },

    renderCell: (item) => {
      return new Date(item.lastUpdated).toLocaleString();
    }
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'actions',
    compare: (a, b) => {
      return a.isComplete ? -1 : 1;
    },
    renderHeaderCell: () => {
      const { t } = useTranslation();

      return t('Actions');
    },
    renderCell: (item) => {
      const { t } = useTranslation();
      const navigate = useNavigate();

      return (
        <TableCellLayout>
          <div className="flex gap-1">
            {
              item.isComplete &&
              <ToolTipButton desc={t('Open Folder')} icon={<Folder20Regular />} onClick={() => {
                OpenFileFolder(`${commonStore.settings.customModelsPath}/${item.name}`, true);
              }} />
            }
            {item.downloadUrl && !item.isComplete &&
              <ToolTipButton desc={t('Download')} icon={<ArrowDownload20Regular />} onClick={() => {
                toastWithButton(`${t('Downloading')} ${item.name}`, t('Check'), () => {
                    navigate({ pathname: '/downloads' });
                  },
                  { autoClose: 3000 });
                AddToDownloadList(`${commonStore.settings.customModelsPath}/${item.name}`, item.downloadUrl!);
              }} />}
            {item.url && <ToolTipButton desc={t('Open Url')} icon={<Open20Regular />} onClick={() => {
              BrowserOpenURL(item.url!);
            }} />}
          </div>
        </TableCellLayout>
      );
    }
  })
];

export const Models: FC = observer(() => {
  const { t } = useTranslation();

  return (
    <Page title={t('Models')} content={
      <div className="flex flex-col gap-2 overflow-hidden">
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center">
            <Text weight="medium">{t('Model Source Manifest List')}</Text>
            <ToolTipButton desc={t('Refresh')} icon={<ArrowClockwise20Regular />} onClick={() => {
              refreshModels(false);
              saveConfigs();
            }} />
          </div>
          <Text size={100}>
            {t('Provide JSON file URLs for the models manifest. Separate URLs with semicolons. The "models" field in JSON files will be parsed into the following table.')}
          </Text>
          <Textarea size="large" resize="vertical"
            value={commonStore.modelSourceManifestList}
            onChange={(e, data) => commonStore.setModelSourceManifestList(data.value)} />
        </div>
        <div className="flex grow overflow-hidden">
          <DataGrid
            items={commonStore.modelSourceList}
            columns={columns}
            sortable={true}
            defaultSortState={{ sortColumn: 'actions', sortDirection: 'ascending' }}
            style={{ display: 'flex' }}
            className="flex-col w-full"
          >
            <DataGridHeader>
              <DataGridRow>
                {({ renderHeaderCell }) => (
                  <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
                )}
              </DataGridRow>
            </DataGridHeader>
            <div className="overflow-y-auto overflow-x-hidden">
              <DataGridBody<ModelSourceItem>>
                {({ item, rowId }) => (
                  (!item.hide || item.isComplete) &&
                  <DataGridRow<ModelSourceItem> key={rowId}>
                    {({ renderCell }) => (
                      <DataGridCell>{renderCell(item)}</DataGridCell>
                    )}
                  </DataGridRow>
                )}
              </DataGridBody>
            </div>
          </DataGrid>
        </div>
      </div>
    } />
  );
});
