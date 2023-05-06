import React, {FC, useEffect} from 'react';
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
import {EditRegular} from '@fluentui/react-icons/lib/fonts';
import {ToolTipButton} from './components/ToolTipButton';
import {ArrowClockwise20Regular} from '@fluentui/react-icons';

type Operation = {
  icon: JSX.Element;
  desc: string
}

type Item = {
  filename: string;
  desc: string;
  size: number;
  lastUpdated: number;
  actions: Operation[];
  isLocal: boolean;
};

const items: Item[] = [
  {
    filename: 'RWKV-4-Raven-14B-v11x-Eng99%-Other1%-20230501-ctx8192.pth',
    desc: 'Mainly English language corpus',
    size: 28297309490,
    lastUpdated: 1,
    actions: [{icon: <EditRegular/>, desc: 'Edit'}],
    isLocal: false
  }
];

const columns: TableColumnDefinition<Item>[] = [
  createTableColumn<Item>({
    columnId: 'file',
    compare: (a, b) => {
      return a.filename.localeCompare(b.filename);
    },
    renderHeaderCell: () => {
      return 'File';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout className="break-all">
          {item.filename}
        </TableCellLayout>
      );
    }
  }),
  createTableColumn<Item>({
    columnId: 'desc',
    compare: (a, b) => {
      return a.desc.localeCompare(b.desc);
    },
    renderHeaderCell: () => {
      return 'Desc';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout>
          {item.desc}
        </TableCellLayout>
      );
    }
  }),
  createTableColumn<Item>({
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
  createTableColumn<Item>({
    columnId: 'lastUpdated',
    compare: (a, b) => {
      return a.lastUpdated - b.lastUpdated;
    },
    renderHeaderCell: () => {
      return 'Last updated';
    },

    renderCell: (item) => {
      return new Date(item.lastUpdated).toLocaleString();
    }
  }),
  createTableColumn<Item>({
    columnId: 'actions',
    compare: (a, b) => {
      return a.isLocal === b.isLocal ? 0 : a.isLocal ? -1 : 1;
    },
    renderHeaderCell: () => {
      return 'Actions';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout>
        </TableCellLayout>
      );
    }
  })
];

export const Models: FC = () => {
  useEffect(() => {
    fetch('https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner/manifest.json')
      .then(
        res => res.json().then(console.log)
      );
  }, []);

  return (
    <div className="flex flex-col box-border gap-5 p-2">
      <Text size={600}>In Development</Text>
      <div className="flex flex-col gap-1">
        <div className="flex justify-between">
          <Text weight="medium">Model Source Url List</Text>
          <ToolTipButton desc="Refresh" icon={<ArrowClockwise20Regular/>}/>
        </div>
        <Text size={100}>description</Text>
        <Textarea size="large" resize="vertical"
                  defaultValue="https://cdn.jsdelivr.net/gh/josstorer/RWKV-Runner/manifest.json;"/>
      </div>
      <DataGrid
        items={items}
        columns={columns}
        sortable={true}
      >
        <DataGridHeader>
          <DataGridRow>
            {({renderHeaderCell}) => (
              <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
            )}
          </DataGridRow>
        </DataGridHeader>
        <DataGridBody<Item>>
          {({item, rowId}) => (
            <DataGridRow<Item> key={rowId}>
              {({renderCell}) => (
                <DataGridCell>{renderCell(item)}</DataGridCell>
              )}
            </DataGridRow>
          )}
        </DataGridBody>
      </DataGrid>
    </div>
  );
};
