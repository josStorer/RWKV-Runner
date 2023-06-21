import React, { FC, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Avatar, Button, Menu, MenuPopover, MenuTrigger, PresenceBadge, Textarea } from '@fluentui/react-components';
import commonStore, { ModelStatus } from '../stores/commonStore';
import { observer } from 'mobx-react-lite';
import { v4 as uuid } from 'uuid';
import classnames from 'classnames';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { KebabHorizontalIcon, PencilIcon, SyncIcon, TrashIcon } from '@primer/octicons-react';
import { ConversationPair } from '../utils/get-conversation-pairs';
import logo from '../assets/images/logo.jpg';
import MarkdownRender from '../components/MarkdownRender';
import { ToolTipButton } from '../components/ToolTipButton';
import { ArrowCircleUp28Regular, Delete28Regular, RecordStop28Regular, Save28Regular } from '@fluentui/react-icons';
import { CopyButton } from '../components/CopyButton';
import { ReadButton } from '../components/ReadButton';
import { toast } from 'react-toastify';
import { WorkHeader } from '../components/WorkHeader';
import { DialogButton } from '../components/DialogButton';
import { OpenFileFolder, OpenSaveFileDialog } from '../../wailsjs/go/backend_golang/App';
import { toastWithButton } from '../utils';

export const userName = 'M E';
export const botName = 'A I';

export const welcomeUuid = 'welcome';

export enum MessageType {
  Normal,
  Error
}

export type Side = 'left' | 'right'

export type Color = 'neutral' | 'brand' | 'colorful'

export type MessageItem = {
  sender: string,
  type: MessageType,
  color: Color,
  avatarImg?: string,
  time: string,
  content: string,
  side: Side,
  done: boolean
}

export type Conversation = {
  [uuid: string]: MessageItem
}

let chatSseController: AbortController | null = null;

const MoreUtilsButton: FC<{ uuid: string, setEditing: (editing: boolean) => void }> = observer(({
  uuid,
  setEditing
}) => {
  const { t } = useTranslation();
  const [speaking, setSpeaking] = useState(false);

  const messageItem = commonStore.conversation[uuid];

  return <Menu>
    <MenuTrigger disableButtonEnhancement>
      <Button icon={<KebabHorizontalIcon />} size="small" appearance="subtle" />
    </MenuTrigger>
    <MenuPopover style={{ minWidth: 0 }}>
      <CopyButton content={messageItem.content} showDelay={500} />
      <ReadButton content={messageItem.content} inSpeaking={speaking} showDelay={500} setSpeakingOuter={setSpeaking} />
      <ToolTipButton desc={t('Edit')} icon={<PencilIcon />} showDelay={500} size="small" appearance="subtle"
        onClick={() => {
          setEditing(true);
        }} />
      <ToolTipButton desc={t('Delete')} icon={<TrashIcon />} showDelay={500} size="small" appearance="subtle"
        onClick={() => {
          commonStore.conversationOrder.splice(commonStore.conversationOrder.indexOf(uuid), 1);
          delete commonStore.conversation[uuid];
        }} />
    </MenuPopover>
  </Menu>;
});

const ChatMessageItem: FC<{
  uuid: string, onSubmit: (message: string | null, answerId: string | null,
    startUuid: string | null, endUuid: string | null, includeEndUuid: boolean) => void
}> = observer(({ uuid, onSubmit }) => {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messageItem = commonStore.conversation[uuid];

  console.log(uuid);

  const setEditingInner = (editing: boolean) => {
    setEditing(editing);
    if (editing) {
      setTimeout(() => {
        const textarea = textareaRef.current;
        if (textarea) {
          textarea.focus();
          textarea.selectionStart = textarea.value.length;
          textarea.selectionEnd = textarea.value.length;
          textarea.style.height = textarea.scrollHeight + 'px';
        }
      });
    }
  };

  return <div
    className={classnames(
      'flex gap-2 mb-2 overflow-hidden',
      messageItem.side === 'left' ? 'flex-row' : 'flex-row-reverse'
    )}
    onMouseEnter={() => {
      const utils = document.getElementById('utils-' + uuid);
      if (utils) utils.classList.remove('invisible');
    }}
    onMouseLeave={() => {
      const utils = document.getElementById('utils-' + uuid);
      if (utils) utils.classList.add('invisible');
    }}
  >
    <Avatar
      color={messageItem.color}
      name={messageItem.sender}
      image={messageItem.avatarImg ? { src: messageItem.avatarImg } : undefined}
    />
    <div
      className={classnames(
        'flex p-2 rounded-lg overflow-hidden',
        editing ? 'grow' : '',
        messageItem.side === 'left' ? 'bg-gray-200' : 'bg-blue-500',
        messageItem.side === 'left' ? 'text-gray-600' : 'text-white'
      )}
    >
      {!editing ?
        <MarkdownRender>{messageItem.content}</MarkdownRender> :
        <Textarea ref={textareaRef}
          className="grow"
          style={{ minWidth: 0 }}
          value={messageItem.content}
          onChange={(e) => {
            messageItem.content = e.target.value;
          }}
          onBlur={() => {
            setEditingInner(false);
          }} />}
    </div>
    <div className="flex flex-col gap-1 items-start">
      <div className="grow" />
      {(messageItem.type === MessageType.Error || !messageItem.done) &&
        <PresenceBadge size="extra-small" status={
          messageItem.type === MessageType.Error ? 'busy' : 'away'
        } />
      }
      <div className="flex invisible" id={'utils-' + uuid}>
        {
          messageItem.sender === botName && uuid !== welcomeUuid &&
          <ToolTipButton desc={t('Retry')} size="small" appearance="subtle"
            icon={<SyncIcon />} onClick={() => {
            onSubmit(null, uuid, null, uuid, false);
          }} />
        }
        <ToolTipButton desc={t('Edit')} icon={<PencilIcon />} size="small" appearance="subtle"
          onClick={() => {
            setEditingInner(true);
          }} />
        <MoreUtilsButton uuid={uuid} setEditing={setEditingInner} />
      </div>
    </div>
  </div>;
});

const ChatPanel: FC = observer(() => {
  const { t } = useTranslation();
  const bodyRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;

  let lastMessageId: string;
  let generating: boolean = false;
  if (commonStore.conversationOrder.length > 0) {
    lastMessageId = commonStore.conversationOrder[commonStore.conversationOrder.length - 1];
    const lastMessage = commonStore.conversation[lastMessageId];
    if (lastMessage.sender === botName)
      generating = !lastMessage.done;
  }

  useEffect(() => {
    if (inputRef.current)
      inputRef.current.style.maxHeight = '16rem';
    scrollToBottom();
  }, []);

  useEffect(() => {
    if (commonStore.conversationOrder.length === 0) {
      commonStore.setConversationOrder([welcomeUuid]);
      commonStore.setConversation({
        [welcomeUuid]: {
          sender: botName,
          type: MessageType.Normal,
          color: 'colorful',
          avatarImg: logo,
          time: new Date().toISOString(),
          content: t('Hello! I\'m RWKV, an open-source and commercially usable large language model.'),
          side: 'left',
          done: true
        }
      });
    }
  }, []);

  const scrollToBottom = () => {
    if (bodyRef.current)
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  };

  const handleKeyDownOrClick = (e: any) => {
    e.stopPropagation();
    if (e.type === 'click' || (e.keyCode === 13 && !e.shiftKey)) {
      e.preventDefault();
      if (commonStore.status.status === ModelStatus.Offline && !commonStore.settings.apiUrl) {
        toast(t('Please click the button in the top right corner to start the model'), { type: 'warning' });
        return;
      }
      if (!commonStore.currentInput) return;
      onSubmit(commonStore.currentInput);
      commonStore.setCurrentInput('');
    }
  };

  // if message is not null, create a user message;
  // if answerId is not null, override the answer with new response;
  // if startUuid is null, start generating api body messages from first message;
  // if endUuid is null, generate api body messages until last message;
  const onSubmit = useCallback((message: string | null = null, answerId: string | null = null,
    startUuid: string | null = null, endUuid: string | null = null, includeEndUuid: boolean = false) => {
    if (message) {
      const newId = uuid();
      commonStore.conversation[newId] = {
        sender: userName,
        type: MessageType.Normal,
        color: 'brand',
        time: new Date().toISOString(),
        content: message,
        side: 'right',
        done: true
      };
      commonStore.setConversation(commonStore.conversation);
      commonStore.conversationOrder.push(newId);
      commonStore.setConversationOrder(commonStore.conversationOrder);
    }

    let startIndex = startUuid ? commonStore.conversationOrder.indexOf(startUuid) : 0;
    let endIndex = endUuid ? (commonStore.conversationOrder.indexOf(endUuid) + (includeEndUuid ? 1 : 0)) : commonStore.conversationOrder.length;
    let targetRange = commonStore.conversationOrder.slice(startIndex, endIndex);

    const messages: ConversationPair[] = [];
    targetRange.forEach((uuid, index) => {
      if (uuid === welcomeUuid)
        return;
      const messageItem = commonStore.conversation[uuid];
      if (messageItem.done && messageItem.type === MessageType.Normal && messageItem.sender === userName) {
        messages.push({ role: 'user', content: messageItem.content });
      } else if (messageItem.done && messageItem.type === MessageType.Normal && messageItem.sender === botName) {
        messages.push({ role: 'assistant', content: messageItem.content });
      }
    });

    if (answerId === null) {
      answerId = uuid();
      commonStore.conversationOrder.push(answerId);
    }
    commonStore.conversation[answerId] = {
      sender: botName,
      type: MessageType.Normal,
      color: 'colorful',
      avatarImg: logo,
      time: new Date().toISOString(),
      content: '',
      side: 'left',
      done: false
    };
    commonStore.setConversation(commonStore.conversation);
    commonStore.setConversationOrder(commonStore.conversationOrder);
    setTimeout(scrollToBottom);
    let answer = '';
    chatSseController = new AbortController();
    fetchEventSource( // https://api.openai.com/v1/chat/completions || http://127.0.0.1:${port}/chat/completions
      commonStore.settings.apiUrl ?
        commonStore.settings.apiUrl + '/v1/chat/completions' :
        `http://127.0.0.1:${port}/chat/completions`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${commonStore.settings.apiKey}`
        },
        body: JSON.stringify({
          messages,
          stream: true,
          model: commonStore.settings.apiChatModelName // 'gpt-3.5-turbo'
        }),
        signal: chatSseController?.signal,
        onmessage(e) {
          scrollToBottom();
          if (e.data === '[DONE]') {
            commonStore.conversation[answerId!].done = true;
            commonStore.conversation[answerId!].content = commonStore.conversation[answerId!].content.trim();
            commonStore.setConversation(commonStore.conversation);
            commonStore.setConversationOrder([...commonStore.conversationOrder]);
            return;
          }
          let data;
          try {
            data = JSON.parse(e.data);
          } catch (error) {
            console.debug('json error', error);
            return;
          }
          if (data.choices && Array.isArray(data.choices) && data.choices.length > 0) {
            answer += data.choices[0]?.delta?.content || '';
            commonStore.conversation[answerId!].content = answer;
            commonStore.setConversation(commonStore.conversation);
            commonStore.setConversationOrder([...commonStore.conversationOrder]);
          }
        },
        async onopen(response) {
          if (response.status !== 200) {
            commonStore.conversation[answerId!].content += '\n[ERROR]\n```\n' + response.statusText + '\n' + (await response.text()) + '\n```';
            commonStore.setConversation(commonStore.conversation);
            commonStore.setConversationOrder([...commonStore.conversationOrder]);
            setTimeout(scrollToBottom);
          }
        },
        onclose() {
          console.log('Connection closed');
        },
        onerror(err) {
          commonStore.conversation[answerId!].type = MessageType.Error;
          commonStore.conversation[answerId!].done = true;
          err = err.message || err;
          if (err && !err.includes('ReadableStreamDefaultReader'))
            commonStore.conversation[answerId!].content += '\n[ERROR]\n```\n' + err + '\n```';
          commonStore.setConversation(commonStore.conversation);
          commonStore.setConversationOrder([...commonStore.conversationOrder]);
          setTimeout(scrollToBottom);
          throw err;
        }
      });
  }, []);

  return (
    <div className="flex flex-col w-full grow gap-4 pt-4 overflow-hidden">
      <div ref={bodyRef} className="grow overflow-y-scroll overflow-x-hidden pr-2">
        {commonStore.conversationOrder.map(uuid =>
          <ChatMessageItem key={uuid} uuid={uuid} onSubmit={onSubmit} />
        )}
      </div>
      <div className="flex items-end gap-2">
        <DialogButton tooltip={t('Clear')}
          icon={<Delete28Regular />}
          size="large" shape="circular" appearance="subtle" title={t('Clear')}
          contentText={t('Are you sure you want to clear the conversation? It cannot be undone.')}
          onConfirm={() => {
            if (generating)
              chatSseController?.abort();
            commonStore.setConversation({});
            commonStore.setConversationOrder([]);
          }} />
        <Textarea
          ref={inputRef}
          className="grow"
          resize="vertical"
          placeholder={t('Type your message here')!}
          value={commonStore.currentInput}
          onChange={(e) => commonStore.setCurrentInput(e.target.value)}
          onKeyDown={handleKeyDownOrClick}
        />
        <ToolTipButton desc={generating ? t('Stop') : t('Send')}
          icon={generating ? <RecordStop28Regular /> : <ArrowCircleUp28Regular />}
          size="large" shape="circular" appearance="subtle"
          onClick={(e) => {
            if (generating) {
              chatSseController?.abort();
              if (lastMessageId) {
                commonStore.conversation[lastMessageId].type = MessageType.Error;
                commonStore.conversation[lastMessageId].done = true;
                commonStore.setConversation(commonStore.conversation);
                commonStore.setConversationOrder([...commonStore.conversationOrder]);
              }
            } else {
              handleKeyDownOrClick(e);
            }
          }} />
        <ToolTipButton desc={t('Save')}
          icon={<Save28Regular />}
          size="large" shape="circular" appearance="subtle"
          onClick={() => {
            let savedContent: string = '';
            commonStore.conversationOrder.forEach((uuid) => {
              const messageItem = commonStore.conversation[uuid];
              savedContent += `**${messageItem.sender}**\n - ${new Date(messageItem.time).toLocaleString()}\n\n${messageItem.content}\n\n`;
            });

            OpenSaveFileDialog('*.md', 'conversation.md', savedContent).then((path) => {
              if (path)
                toastWithButton(t('Conversation Saved'), t('Open'), () => {
                  OpenFileFolder(path, false);
                });
            }).catch(e => {
              toast(t('Error') + ' - ' + e.message || e, { type: 'error', autoClose: 2500 });
            });
          }} />
      </div>
    </div>
  );
});

export const Chat: FC = observer(() => {
  return (
    <div className="flex flex-col gap-1 p-2 h-full overflow-hidden">
      <WorkHeader />
      <ChatPanel />
    </div>
  );
});
