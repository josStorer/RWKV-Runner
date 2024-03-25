import React, { FC, useState } from 'react';
import { DragDropContext, Draggable, Droppable, DropResult } from 'react-beautiful-dnd';
import commonStore from '../../stores/commonStore';
import { observer } from 'mobx-react-lite';
import { v4 as uuid } from 'uuid';
import { Card, Dropdown, Option, Textarea } from '@fluentui/react-components';
import { useTranslation } from 'react-i18next';
import { ToolTipButton } from '../../components/ToolTipButton';
import { Delete20Regular, ReOrderDotsVertical20Regular } from '@fluentui/react-icons';
import { Preset } from '../../types/presets';
import { ConversationMessage, Role } from '../../types/chat';

type Item = {
  id: string;
  role: Role;
  content: string;
}

const getItems = (messages: ConversationMessage[]) =>
  messages.map((message, index) => ({
    id: uuid(),
    role: message.role,
    content: message.content
  })) as Item[];

const reorder = (list: Item[], startIndex: number, endIndex: number) => {
  const result = Array.from(list);
  const [removed] = result.splice(startIndex, 1);
  result.splice(endIndex, 0, removed);

  return result;
};

const MessagesEditor: FC = observer(() => {
  const { t } = useTranslation();

  const editingPreset = commonStore.editingPreset!;
  const setEditingPreset = (newParams: Partial<Preset>) => {
    commonStore.setEditingPreset({
      ...editingPreset,
      ...newParams
    });
  };

  const [items, setItems] = useState(getItems(editingPreset.messages));

  const updateItems = (items: Item[]) => {
    setEditingPreset({
      messages: items.map(item => ({
        role: item.role,
        content: item.content
      }))
    });
    setItems(items);
  };

  const onDragEnd = (result: DropResult) => {
    if (!result.destination) {
      return;
    }

    const newItems = reorder(
      items,
      result.source.index,
      result.destination.index
    );

    updateItems(newItems);
  };

  const createNewItem = () => {
    const newItems: Item[] = [...items, {
      id: uuid(),
      role: 'assistant',
      content: ''
    }];
    updateItems(newItems);
  };

  const deleteItem = (id: string) => {
    const newItems: Item[] = items.filter(item => item.id !== id);
    updateItems(newItems);
  };

  return (
    <div className="grid grid-cols-1 gap-2 overflow-hidden">
      <ToolTipButton text={t('New')}
        desc={t('Create a new user or AI message content. You can prepare a chat record with AI here, and fill in the responses you want to get from AI in the tone of AI. When you use this preset, the chat record will be processed, and at this point, AI will better understand what you want it to do or what role to play.')}
        style={{ width: '100%' }}
        onClick={createNewItem} />
      <div className="overflow-x-hidden overflow-y-auto p-2">
        <DragDropContext onDragEnd={onDragEnd}>
          <Droppable droppableId="droppable">
            {(provided, snapshot) => (
              <div
                {...provided.droppableProps}
                ref={provided.innerRef}
              >
                {items.map((item, index) => (
                  <Draggable key={item.id} draggableId={item.id} index={index}>
                    {(provided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                        style={provided.draggableProps.style}
                        className="select-none mb-2"
                      >
                        <div className="flex">
                          <Card appearance="outline"
                            style={{ borderTopRightRadius: 0, borderBottomRightRadius: 0 }}>
                            <ReOrderDotsVertical20Regular />
                          </Card>
                          <Dropdown style={{ minWidth: 0, borderRadius: 0 }}
                            listbox={{ style: { minWidth: 'fit-content' } }}
                            value={t(item.role)!}
                            selectedOptions={[item.role]}
                            onOptionSelect={(_, data) => {
                              if (data.optionValue) {
                                items[index] = {
                                  ...item,
                                  role: data.optionValue as Role
                                };
                                updateItems([...items]);
                              }
                            }}>
                            <Option value="user">{t('user')!}</Option>
                            <Option value="assistant">{t('assistant')!}</Option>
                            <Option value="system">{t('system')!}</Option>
                          </Dropdown>
                          <Textarea resize="vertical" className="grow" value={item.content}
                            style={{ minWidth: 0, borderRadius: 0 }}
                            onChange={(e, data) => {
                              items[index] = {
                                ...item,
                                content: data.value
                              };
                              updateItems([...items]);
                            }}></Textarea>
                          <ToolTipButton
                            style={{ borderTopLeftRadius: 0, borderBottomLeftRadius: 0 }} desc={t('Delete')}
                            icon={<Delete20Regular />} onClick={() => {
                            deleteItem(item.id);
                          }} />
                        </div>
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>
      </div>
    </div>
  );
});

export default MessagesEditor;
