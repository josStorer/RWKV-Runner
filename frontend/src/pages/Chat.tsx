import React, { FC, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Avatar, PresenceBadge, Textarea } from '@fluentui/react-components';
import commonStore, { ModelStatus } from '../stores/commonStore';
import { observer } from 'mobx-react-lite';
import { v4 as uuid } from 'uuid';
import classnames from 'classnames';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { ConversationPair, getConversationPairs, Record } from '../utils/get-conversation-pairs';
import logo from '../assets/images/logo.jpg';
import MarkdownRender from '../components/MarkdownRender';
import { ToolTipButton } from '../components/ToolTipButton';
import { ArrowCircleUp28Regular, Delete28Regular, RecordStop28Regular } from '@fluentui/react-icons';
import { CopyButton } from '../components/CopyButton';
import { ReadButton } from '../components/ReadButton';
import { toast } from 'react-toastify';
import { WorkHeader } from '../components/WorkHeader';

export const userName = 'M E';
export const botName = 'A I';

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

export type Conversations = {
  [uuid: string]: MessageItem
}

let chatSseController: AbortController | null = null;

const ChatPanel: FC = observer(() => {
  const { t } = useTranslation();
  const bodyRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;

  let lastMessageId: string;
  let generating: boolean = false;
  if (commonStore.conversationsOrder.length > 0) {
    lastMessageId = commonStore.conversationsOrder[commonStore.conversationsOrder.length - 1];
    const lastMessage = commonStore.conversations[lastMessageId];
    if (lastMessage.sender === botName)
      generating = !lastMessage.done;
  }

  useEffect(() => {
    if (inputRef.current)
      inputRef.current.style.maxHeight = '16rem';
    scrollToBottom();
  }, []);

  useEffect(() => {
    if (commonStore.conversationsOrder.length === 0) {
      commonStore.setConversationsOrder(['welcome']);
      commonStore.setConversations({
        'welcome': {
          sender: botName,
          type: MessageType.Normal,
          color: 'colorful',
          avatarImg: logo,
          time: new Date().toISOString(),
          content: t('Hello! I\'m RWKV, an open-source and commercially available large language model.'),
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
      if (commonStore.status.status === ModelStatus.Offline) {
        toast(t('Please click the button in the top right corner to start the model'), { type: 'warning' });
        return;
      }
      if (!commonStore.currentInput) return;
      onSubmit(commonStore.currentInput);
      commonStore.setCurrentInput('');
    }
  };

  const onSubmit = (message: string) => {
    const newId = uuid();
    commonStore.conversations[newId] = {
      sender: userName,
      type: MessageType.Normal,
      color: 'brand',
      time: new Date().toISOString(),
      content: message,
      side: 'right',
      done: true
    };
    commonStore.setConversations(commonStore.conversations);
    commonStore.conversationsOrder.push(newId);
    commonStore.setConversationsOrder(commonStore.conversationsOrder);

    const records: Record[] = [];
    commonStore.conversationsOrder.forEach((uuid, index) => {
      const conversation = commonStore.conversations[uuid];
      if (conversation.done && conversation.type === MessageType.Normal && conversation.sender === botName) {
        if (index > 0) {
          const questionId = commonStore.conversationsOrder[index - 1];
          const question = commonStore.conversations[questionId];
          if (question.done && question.type === MessageType.Normal && question.sender === userName) {
            records.push({ question: question.content, answer: conversation.content });
          }
        }
      }
    });
    const messages = getConversationPairs(records, false);
    (messages as ConversationPair[]).push({ role: 'user', content: message });

    const answerId = uuid();
    commonStore.conversations[answerId] = {
      sender: botName,
      type: MessageType.Normal,
      color: 'colorful',
      avatarImg: logo,
      time: new Date().toISOString(),
      content: '',
      side: 'left',
      done: false
    };
    commonStore.setConversations(commonStore.conversations);
    commonStore.conversationsOrder.push(answerId);
    commonStore.setConversationsOrder(commonStore.conversationsOrder);
    setTimeout(scrollToBottom);
    let answer = '';
    chatSseController = new AbortController();
    fetchEventSource(`http://127.0.0.1:${port}/chat/completions`, // https://api.openai.com/v1/chat/completions || http://127.0.0.1:${port}/chat/completions
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer sk-`
        },
        body: JSON.stringify({
          messages,
          stream: true,
          model: 'gpt-3.5-turbo'
        }),
        signal: chatSseController?.signal,
        onmessage(e) {
          console.log('sse message', e);
          scrollToBottom();
          if (e.data === '[DONE]') {
            commonStore.conversations[answerId].done = true;
            commonStore.conversations[answerId].content = commonStore.conversations[answerId].content.trim();
            commonStore.setConversations(commonStore.conversations);
            commonStore.setConversationsOrder([...commonStore.conversationsOrder]);
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
            commonStore.conversations[answerId].content = answer;
            commonStore.setConversations(commonStore.conversations);
            commonStore.setConversationsOrder([...commonStore.conversationsOrder]);
          }
        },
        onclose() {
          console.log('Connection closed');
        },
        onerror(err) {
          commonStore.conversations[answerId].type = MessageType.Error;
          commonStore.conversations[answerId].done = true;
          commonStore.setConversations(commonStore.conversations);
          commonStore.setConversationsOrder([...commonStore.conversationsOrder]);
          throw err;
        }
      });
  };

  return (
    <div className="flex flex-col w-full grow gap-4 pt-4 overflow-hidden">
      <div ref={bodyRef} className="grow overflow-y-scroll overflow-x-hidden pr-2">
        {commonStore.conversationsOrder.map((uuid, index) => {
          const conversation = commonStore.conversations[uuid];
          return <div
            key={uuid}
            className={classnames(
              'flex gap-2 mb-2 overflow-hidden',
              conversation.side === 'left' ? 'flex-row' : 'flex-row-reverse'
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
              color={conversation.color}
              name={conversation.sender}
              image={conversation.avatarImg ? { src: conversation.avatarImg } : undefined}
            />
            <div
              className={classnames(
                'p-2 rounded-lg overflow-hidden',
                conversation.side === 'left' ? 'bg-gray-200' : 'bg-blue-500',
                conversation.side === 'left' ? 'text-gray-600' : 'text-white'
              )}
            >
              <MarkdownRender>{conversation.content}</MarkdownRender>
            </div>
            <div className="flex flex-col gap-1 items-start">
              <div className="grow" />
              {(conversation.type === MessageType.Error || !conversation.done) &&
                <PresenceBadge size="extra-small" status={
                  conversation.type === MessageType.Error ? 'busy' : 'away'
                } />
              }
              <div className="flex invisible" id={'utils-' + uuid}>
                <ReadButton content={conversation.content} />
                <CopyButton content={conversation.content} />
              </div>
            </div>
          </div>;
        })}
      </div>
      <div className="flex items-end gap-2">
        <ToolTipButton desc={t('Clear')}
          icon={<Delete28Regular />}
          size="large" shape="circular" appearance="subtle"
          onClick={(e) => {
            if (generating)
              chatSseController?.abort();
            commonStore.setConversations({});
            commonStore.setConversationsOrder([]);
          }}
        />
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
                commonStore.conversations[lastMessageId].type = MessageType.Error;
                commonStore.conversations[lastMessageId].done = true;
                commonStore.setConversations(commonStore.conversations);
                commonStore.setConversationsOrder([...commonStore.conversationsOrder]);
              }
            } else {
              handleKeyDownOrClick(e);
            }
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
