import React, {FC} from 'react';
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
import {ToolTipButton} from '../components/ToolTipButton';
import {ArrowClockwise20Regular, ArrowDownload20Regular, Open20Regular} from '@fluentui/react-icons';
import {observer} from 'mobx-react-lite';
import commonStore, {ModelSourceItem} from '../stores/commonStore';
import {BrowserOpenURL} from '../../wailsjs/runtime';
import {DownloadFile} from '../../wailsjs/go/backend_golang/App';

const columns: TableColumnDefinition<ModelSourceItem>[] = [
  createTableColumn<ModelSourceItem>({
    columnId: 'file',
    compare: (a, b) => {
      return a.name.localeCompare(b.name);
    },
    renderHeaderCell: () => {
      return 'File';
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
      return a.desc['en'].localeCompare(b.desc['en']);
    },
    renderHeaderCell: () => {
      return 'Desc';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout>
          {item.desc['en']}
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
      return 'Size';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout>
          {(item.size / (1024 * 1024 * 1024)).toFixed(2) + 'GB'}
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
      return a.lastUpdatedMs - b.lastUpdatedMs;
    },
    renderHeaderCell: () => {
      return 'Last updated';
    },

    renderCell: (item) => {
      return new Date(item.lastUpdated).toLocaleString();
    }
  }),
  createTableColumn<ModelSourceItem>({
    columnId: 'actions',
    compare: (a, b) => {
      return a.isDownloading ? 0 : a.isLocal ? 1 : 2;
    },
    renderHeaderCell: () => {
      return 'Actions';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout>
          <div className="flex gap-1">
            <ToolTipButton desc="Download" icon={<ArrowDownload20Regular/>} onClick={() => {
              DownloadFile(`./models/${item.name}`, item.downloadUrl);
            }}/>
            <ToolTipButton desc="Open Url" icon={<Open20Regular/>} onClick={() => {
              BrowserOpenURL(item.url);
            }}/>
          </div>
        </TableCellLayout>
      );
    }
  })
];

export const Models: FC = observer(() => {
  return (
    <div className="flex flex-col gap-2 p-2 h-full">
      <Text size={600}>Models</Text>
      <div className="flex flex-col gap-1">
        <div className="flex justify-between">
          <Text weight="medium">Model Source Manifest List</Text>
          <ToolTipButton desc="Refresh" icon={<ArrowClockwise20Regular/>}/>
        </div>
        <Text size={100}>Provide JSON file URLs for the models manifest. Separate URLs with semicolons. The "models"
          field in JSON files will be parsed into the following table.</Text>
        <Textarea size="large" resize="vertical"
                  defaultValue={commonStore.modelSourceManifestList}
                  onChange={(e, data) => commonStore.setModelSourceManifestList(data.value)}/>
      </div>
      <div className="flex grow overflow-hidden">
        <DataGrid
          items={commonStore.modelSourceList}
          columns={columns}
          sortable={true}
          style={{display: 'flex'}}
          className="flex-col"
        >
          <DataGridHeader>
            <DataGridRow>
              {({renderHeaderCell}) => (
                <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
              )}
            </DataGridRow>
          </DataGridHeader>
          <div className="overflow-y-auto overflow-x-hidden">
            <DataGridBody<ModelSourceItem>>
              {({item, rowId}) => (
                <DataGridRow<ModelSourceItem> key={rowId}>
                  {({renderCell}) => (
                    <DataGridCell>{renderCell(item)}</DataGridCell>
                  )}
                </DataGridRow>
              )}
            </DataGridBody>
          </div>
        </DataGrid>
      </div>
    </div>
  );
});
