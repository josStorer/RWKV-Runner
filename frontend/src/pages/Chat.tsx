import React, {FC, useEffect, useRef, useState} from 'react';
import {useTranslation} from 'react-i18next';
import {RunButton} from '../components/RunButton';
import {Avatar, Divider, PresenceBadge, Text, Textarea} from '@fluentui/react-components';
import commonStore, {ModelStatus} from '../stores/commonStore';
import {observer} from 'mobx-react-lite';
import {PresenceBadgeStatus} from '@fluentui/react-badge';
import {ConfigSelector} from '../components/ConfigSelector';
import {v4 as uuid} from 'uuid';
import classnames from 'classnames';
import {fetchEventSource} from '@microsoft/fetch-event-source';
import {ConversationPair, getConversationPairs, Record} from '../utils/get-conversation-pairs';
import logo from '../../../build/appicon.png';
import MarkdownRender from '../components/MarkdownRender';
import {ToolTipButton} from '../components/ToolTipButton';
import {ArrowCircleUp28Regular, Delete28Regular, RecordStop28Regular} from '@fluentui/react-icons';
import {CopyButton} from '../components/CopyButton';
import {ReadButton} from '../components/ReadButton';

const userName = 'M E';
const botName = 'A I';

enum MessageType {
  Normal,
  Error
}

type Side = 'left' | 'right'

type Color = 'neutral' | 'brand' | 'colorful'

type MessageItem = {
  sender: string,
  type: MessageType,
  color: Color,
  avatarImg?: string,
  time: string,
  content: string,
  side: Side,
  done: boolean
}

type Conversations = {
  [uuid: string]: MessageItem
}

const ChatPanel: FC = observer(() => {
  const {t} = useTranslation();
  const [message, setMessage] = useState('');
  const [conversations, setConversations] = useState<Conversations>({
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
  const [conversationsOrder, setConversationsOrder] = useState<string[]>(['welcome']);
  const bodyRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;
  const sseControllerRef = useRef<AbortController | null>(null);

  let lastMessageId: string;
  let generating: boolean = false;
  if (conversationsOrder.length > 0) {
    lastMessageId = conversationsOrder[conversationsOrder.length - 1];
    const lastMessage = conversations[lastMessageId];
    if (lastMessage.sender === botName)
      generating = !lastMessage.done;
  }

  useEffect(() => {
    if (inputRef.current)
      inputRef.current.style.maxHeight = '16rem';
  }, []);

  const scrollToBottom = () => {
    if (bodyRef.current)
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  };

  const handleKeyDownOrClick = (e: any) => {
    e.stopPropagation();
    if (e.type === 'click' || (e.keyCode === 13 && !e.shiftKey)) {
      e.preventDefault();
      if (!message) return;
      onSubmit(message);
      setMessage('');
    }
  };

  const onSubmit = (message: string) => {
    const newId = uuid();
    conversations[newId] = {
      sender: userName,
      type: MessageType.Normal,
      color: 'brand',
      time: new Date().toISOString(),
      content: message,
      side: 'right',
      done: true
    };
    setConversations(conversations);
    conversationsOrder.push(newId);
    setConversationsOrder(conversationsOrder);

    const records: Record[] = [];
    conversationsOrder.forEach((uuid, index) => {
      const conversation = conversations[uuid];
      if (conversation.done && conversation.type === MessageType.Normal && conversation.sender === botName) {
        if (index > 0) {
          const questionId = conversationsOrder[index - 1];
          const question = conversations[questionId];
          if (question.done && question.type === MessageType.Normal && question.sender === userName) {
            records.push({question: question.content, answer: conversation.content});
          }
        }
      }
    });
    const messages = getConversationPairs(records, false);
    (messages as ConversationPair[]).push({role: 'user', content: message});

    const answerId = uuid();
    conversations[answerId] = {
      sender: botName,
      type: MessageType.Normal,
      color: 'colorful',
      avatarImg: logo,
      time: new Date().toISOString(),
      content: '',
      side: 'left',
      done: false
    };
    setConversations(conversations);
    conversationsOrder.push(answerId);
    setConversationsOrder(conversationsOrder);
    setTimeout(scrollToBottom);
    let answer = '';
    sseControllerRef.current = new AbortController();
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
        signal: sseControllerRef.current?.signal,
        onmessage(e) {
          console.log('sse message', e);
          scrollToBottom();
          if (e.data === '[DONE]') {
            conversations[answerId].done = true;
            setConversations(conversations);
            setConversationsOrder([...conversationsOrder]);
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
            conversations[answerId].content = answer;
            setConversations(conversations);
            setConversationsOrder([...conversationsOrder]);
          }
        },
        onclose() {
          console.log('Connection closed');
        },
        onerror(err) {
          conversations[answerId].type = MessageType.Error;
          conversations[answerId].done = true;
          setConversations(conversations);
          setConversationsOrder([...conversationsOrder]);
          throw err;
        }
      });
  };

  return (
    <div className="flex flex-col w-full grow gap-4 pt-4 overflow-hidden">
      <div ref={bodyRef} className="grow overflow-y-scroll overflow-x-hidden pr-2">
        {conversationsOrder.map((uuid, index) => {
          const conversation = conversations[uuid];
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
              image={conversation.avatarImg ? {src: conversation.avatarImg} : undefined}
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
              <div className="grow"/>
              {(conversation.type === MessageType.Error || !conversation.done) &&
                <PresenceBadge size="extra-small" status={
                  conversation.type === MessageType.Error ? 'busy' : 'away'
                }/>
              }
              <div className="flex invisible" id={'utils-' + uuid}>
                <ReadButton content={conversation.content}/>
                <CopyButton content={conversation.content}/>
              </div>
            </div>
          </div>;
        })}
      </div>
      <div className="flex items-end gap-2">
        <ToolTipButton desc={t('Clear')}
                       icon={<Delete28Regular/>}
                       size="large" shape="circular" appearance="subtle"
                       onClick={(e) => {
                         setConversations({});
                         setConversationsOrder([]);
                       }}
        />
        <Textarea
          ref={inputRef}
          className="grow"
          resize="vertical"
          placeholder={t('Type your message here')!}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDownOrClick}
        />
        <ToolTipButton desc={generating ? t('Stop') : t('Send')}
                       icon={generating ? <RecordStop28Regular/> : <ArrowCircleUp28Regular/>}
                       size="large" shape="circular" appearance="subtle"
                       onClick={(e) => {
                         if (generating) {
                           sseControllerRef.current?.abort();
                           if (lastMessageId) {
                             conversations[lastMessageId].type = MessageType.Error;
                             conversations[lastMessageId].done = true;
                             setConversations(conversations);
                             setConversationsOrder([...conversationsOrder]);
                           }
                         } else {
                           handleKeyDownOrClick(e);
                         }
                       }}/>
      </div>
    </div>
  );
});

const statusText = {
  [ModelStatus.Offline]: 'Offline',
  [ModelStatus.Starting]: 'Starting',
  [ModelStatus.Loading]: 'Loading',
  [ModelStatus.Working]: 'Working'
};

const badgeStatus: { [modelStatus: number]: PresenceBadgeStatus } = {
  [ModelStatus.Offline]: 'unknown',
  [ModelStatus.Starting]: 'away',
  [ModelStatus.Loading]: 'away',
  [ModelStatus.Working]: 'available'
};

export const Chat: FC = observer(() => {
  const {t} = useTranslation();
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;

  return (
    <div className="flex flex-col gap-1 p-2 h-full overflow-hidden">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <PresenceBadge status={badgeStatus[commonStore.modelStatus]}/>
          <Text size={100}>{t('Model Status') + ': ' + t(statusText[commonStore.modelStatus])}</Text>
        </div>
        <div className="flex items-center gap-2">
          <ConfigSelector size="small"/>
          <RunButton iconMode/>
        </div>
      </div>
      <Text size={100}>
        {t('This tool’s API is compatible with OpenAI API. It can be used with any ChatGPT tool you like. Go to the settings of some ChatGPT tool, replace the \'https://api.openai.com\' part in the API address with \'') + `http://127.0.0.1:${port}` + '\'.'}
      </Text>
      <Divider style={{flexGrow: 0}}/>
      <ChatPanel/>
    </div>
  );
});
