import {
  Avatar,
  Button,
  createTableColumn,
  Dropdown,
  Input,
  PresenceBadgeStatus,
  Select,
  Slider,
  Switch,
  TableCellLayout,
  TableColumnDefinition,
  Text,
  Tooltip
} from '@fluentui/react-components';
import {
  AddCircle20Regular,
  Delete20Regular,
  DocumentPdfRegular,
  DocumentRegular,
  EditRegular,
  FolderRegular,
  OpenRegular,
  PeopleRegular,
  Save20Regular,
  VideoRegular
} from '@fluentui/react-icons';
import React, {FC} from 'react';
import {Section} from './components/Section';
import {Labeled} from './components/Labeled';
import {ToolTipButton} from './components/ToolTipButton';

type FileCell = {
  label: string;
  icon: JSX.Element;
};

type LastUpdatedCell = {
  label: string;
  timestamp: number;
};

type LastUpdateCell = {
  label: string;
  icon: JSX.Element;
};

type AuthorCell = {
  label: string;
  status: PresenceBadgeStatus;
};

type Item = {
  file: FileCell;
  author: AuthorCell;
  lastUpdated: LastUpdatedCell;
  lastUpdate: LastUpdateCell;
};

const items: Item[] = [
  {
    file: {label: 'Meeting notes', icon: <DocumentRegular/>},
    author: {label: 'Max Mustermann', status: 'available'},
    lastUpdated: {label: '7h ago', timestamp: 1},
    lastUpdate: {
      label: 'You edited this',
      icon: <EditRegular/>
    }
  },
  {
    file: {label: 'Thursday presentation', icon: <FolderRegular/>},
    author: {label: 'Erika Mustermann', status: 'busy'},
    lastUpdated: {label: 'Yesterday at 1:45 PM', timestamp: 2},
    lastUpdate: {
      label: 'You recently opened this',
      icon: <OpenRegular/>
    }
  },
  {
    file: {label: 'Training recording', icon: <VideoRegular/>},
    author: {label: 'John Doe', status: 'away'},
    lastUpdated: {label: 'Yesterday at 1:45 PM', timestamp: 2},
    lastUpdate: {
      label: 'You recently opened this',
      icon: <OpenRegular/>
    }
  },
  {
    file: {label: 'Purchase order', icon: <DocumentPdfRegular/>},
    author: {label: 'Jane Doe', status: 'offline'},
    lastUpdated: {label: 'Tue at 9:30 AM', timestamp: 3},
    lastUpdate: {
      label: 'You shared this in a Teams chat',
      icon: <PeopleRegular/>
    }
  }
];

const columns: TableColumnDefinition<Item>[] = [
  createTableColumn<Item>({
    columnId: 'file',
    compare: (a, b) => {
      return a.file.label.localeCompare(b.file.label);
    },
    renderHeaderCell: () => {
      return 'File';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout media={item.file.icon}>
          {item.file.label}
        </TableCellLayout>
      );
    }
  }),
  createTableColumn<Item>({
    columnId: 'author',
    compare: (a, b) => {
      return a.author.label.localeCompare(b.author.label);
    },
    renderHeaderCell: () => {
      return 'Author';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout
          media={
            <Avatar
              aria-label={item.author.label}
              name={item.author.label}
              badge={{status: item.author.status}}
            />
          }
        >
          {item.author.label}
        </TableCellLayout>
      );
    }
  }),
  createTableColumn<Item>({
    columnId: 'lastUpdated',
    compare: (a, b) => {
      return a.lastUpdated.timestamp - b.lastUpdated.timestamp;
    },
    renderHeaderCell: () => {
      return 'Last updated';
    },

    renderCell: (item) => {
      return item.lastUpdated.label;
    }
  }),
  createTableColumn<Item>({
    columnId: 'lastUpdate',
    compare: (a, b) => {
      return a.lastUpdate.label.localeCompare(b.lastUpdate.label);
    },
    renderHeaderCell: () => {
      return 'Last update';
    },
    renderCell: (item) => {
      return (
        <TableCellLayout media={item.lastUpdate.icon}>
          {item.lastUpdate.label}
        </TableCellLayout>
      );
    }
  })
];

// <DataGrid
//   items={items}
//   columns={columns}
// >
//   <DataGridBody<Item>>
//     {({ item, rowId }) => (
//       <DataGridRow<Item> key={rowId}>
//         {({ renderCell }) => (
//           <DataGridCell>{renderCell(item)}</DataGridCell>
//         )}
//       </DataGridRow>
//     )}
//   </DataGridBody>
// </DataGrid>

export const Configs: FC = () => {
  return (
    <div className="flex flex-col box-border gap-5 p-2">
      <Text size={600}>Configs</Text>
      <Section
        title="Config List"
        content={
          <div className="flex gap-5 items-center w-full">
            <Dropdown className="w-full"/>
            <ToolTipButton desc="New Config" icon={<AddCircle20Regular/>}/>
            <ToolTipButton desc="Delete Config" icon={<Delete20Regular/>}/>
            <ToolTipButton desc="Save Config" icon={<Save20Regular/>}/>
          </div>
        }
      />
      <Section
        title="Default API Parameters"
        desc="Hover your mouse over the text to view a detailed description. Settings marked with * will take effect immediately after being saved."
        content={
          <div className="flex flex-col gap-1">
            <div className="grid grid-cols-2">
              <Labeled label="API Port" desc="127.0.0.1:8000" content={
                <Input type="number" min={1} max={65535} step={1}/>
              }/>
              <Labeled label="Max Response Token *" content={
                <div className="flex items-center">
                  <Slider className="w-48" step={400} min={100} max={8100}/>
                  <Text>1000</Text>
                </div>
              }/>
            </div>
            <div className="grid grid-cols-2">
              <Labeled label="Temperature *" content={
                <Slider/>
              }/>
              <Labeled label="Top_P *" content={
                <Slider/>
              }/>
            </div>
            <div className="grid grid-cols-2">
              <Labeled label="Presence Penalty *" content={
                <Slider/>
              }/>
              <Labeled label="Count Penalty *" content={
                <Slider/>
              }/>
            </div>
          </div>
        }
      />
      <Section
        title="Model Parameters"
        content={
          <div className="flex flex-col gap-1">
            <div className="grid grid-cols-2">
              <Labeled label="Device" content={
                <Select className="w-28">
                  <option>CPU</option>
                  <option>CUDA: 0</option>
                </Select>
              }/>
              <Labeled label="Precision" content={
                <Select className="w-28">
                  <option>fp16</option>
                  <option>int8</option>
                  <option>fp32</option>
                </Select>
              }/>
            </div>
            <div className="grid grid-cols-2">
              <Labeled label="Streamed Layers" content={
                <Slider/>
              }/>
              <Labeled label="Enable High Precision For Last Layer" content={
                <Switch/>
              }/>
            </div>
          </div>
        }
      />
      <div className="fixed bottom-2 right-2">
        <Button appearance="primary" size="large">Run</Button>
      </div>
    </div>
  );
};
