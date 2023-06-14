import React, { FC, useEffect, useRef } from 'react';
import { observer } from 'mobx-react-lite';
import { WorkHeader } from '../components/WorkHeader';
import { Button, Dropdown, Input, Option, Textarea } from '@fluentui/react-components';
import { Labeled } from '../components/Labeled';
import { ValuedSlider } from '../components/ValuedSlider';
import { useTranslation } from 'react-i18next';
import { ApiParameters } from './Configs';
import commonStore, { ModelStatus } from '../stores/commonStore';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { toast } from 'react-toastify';

export type CompletionParams = Omit<ApiParameters, 'apiPort'> & {
  stop: string,
  injectStart: string,
  injectEnd: string
};

export type CompletionPreset = {
  name: string,
  prompt: string,
  params: CompletionParams
}

export const defaultPresets: CompletionPreset[] = [{
  name: 'Writer',
  prompt: 'The following is an epic science fiction masterpiece that is immortalized, with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n',
  params: {
    maxResponseToken: 500,
    temperature: 1.2,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: '\\n\\nBob',
    injectStart: '',
    injectEnd: ''
  }
}, {
  name: 'Translator',
  prompt: 'Translate this into Chinese.\n\nEnglish: What rooms do you have available?',
  params: {
    maxResponseToken: 500,
    temperature: 1,
    topP: 0.3,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: '\\nEnglish',
    injectStart: '\\nChinese: ',
    injectEnd: '\\nEnglish: '
  }
}, {
  name: 'Catgirl',
  prompt: 'The following is a conversation between a cat girl and her owner. The cat girl is a humanized creature that behaves like a cat but is humanoid. At the end of each sentence in the dialogue, she will add \"Meow~\". In the following content, Bob represents the owner and Alice represents the cat girl.\n\nBob: Hello.\n\nAlice: I\'m here, meow~.\n\nBob: Can you tell jokes?',
  params: {
    maxResponseToken: 500,
    temperature: 1.2,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: '\\n\\nBob',
    injectStart: '\\n\\nAlice: ',
    injectEnd: '\\n\\nBob: '
  }
}, {
  name: 'Chinese Kongfu',
  prompt: 'Bob: 请你扮演一个文本冒险游戏，我是游戏主角。这是一个玄幻修真世界，有四大门派。我输入我的行动，请你显示行动结果，并具体描述环境。我的第一个行动是“醒来”，请开始故事。',
  params: {
    maxResponseToken: 500,
    temperature: 1.1,
    topP: 0.7,
    presencePenalty: 0.3,
    frequencyPenalty: 0.3,
    stop: '\\n\\nBob',
    injectStart: '\\n\\nAlice: ',
    injectEnd: '\\n\\nBob: '
  }
}, {
// }, {
//   name: 'Explain Code',
//   prompt: 'export async function startup() {\n  FileExists(\'cache.json\').then((exists) => {\n    if (exists)\n      downloadProgramFiles();\n    else {\n      deleteDynamicProgramFiles().then(downloadProgramFiles);\n    }\n  });\n  EventsOn(\'downloadList\', (data) => {\n    if (data)\n      commonStore.setDownloadList(data);\n  });\n\n  initCache().then(initRemoteText);\n\n  await initConfig();\n\n  if (commonStore.settings.autoUpdatesCheck) // depends on config settings\n    checkUpdate();\n\n  getStatus(1000).then(status => { // depends on config api port\n    if (status)\n      commonStore.setStatus(status);\n  });\n}\n\n\"\"\"\nHere\'s what the above code is doing, explained in a concise way:\n',
//   params: {
//     maxResponseToken: 500,
//     temperature: 0.8,
//     topP: 0.7,
//     presencePenalty: 0.4,
//     frequencyPenalty: 0.4,
//     stop: '\\n\\n',
//     injectStart: '',
//     injectEnd: ''
//   }
// }, {
  name: 'Werewolf',
  prompt: 'There is currently a game of Werewolf with six players, including a Seer (who can check identities at night), two Werewolves (who can choose someone to kill at night), a Bodyguard (who can choose someone to protect at night), two Villagers (with no special abilities), and a game host. Bob will play as Player 1, Alice will play as Players 2-6 and the game host, and they will begin playing together. Every night, the host will ask Bob for his action and simulate the actions of the other players. During the day, the host will oversee the voting process and ask Bob for his vote. \n\nAlice: Next, I will act as the game host and assign everyone their roles, including randomly assigning yours. Then, I will simulate the actions of Players 2-6 and let you know what happens each day. Based on your assigned role, you can tell me your actions and I will let you know the corresponding results each day.\n\nBob: Okay, I understand. Let\'s begin. Please assign me a role. Am I the Seer, Werewolf, Villager, or Bodyguard?\n\nAlice: You are the Seer. Now that night has fallen, please choose a player to check his identity.\n\nBob: Tonight, I want to check Player 2 and find out his role.',
  params: {
    maxResponseToken: 500,
    temperature: 1.2,
    topP: 0.4,
    presencePenalty: 0.5,
    frequencyPenalty: 0.5,
    stop: '\\n\\nBob',
    injectStart: '\\n\\nAlice: ',
    injectEnd: '\\n\\nBob: '
  }
}, {
  name: 'Instruction',
  prompt: 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n# Instruction:\nWrite a story using the following information\n\n# Input:\nA man named Alex chops a tree down\n\n# Response:\n',
  params: {
    maxResponseToken: 500,
    temperature: 1.2,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: '',
    injectStart: '',
    injectEnd: ''
  }
}, {
  name: 'Blank',
  prompt: '',
  params: {
    maxResponseToken: 500,
    temperature: 1,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: '',
    injectStart: '',
    injectEnd: ''
  }
}];

let completionSseController: AbortController | null = null;

const CompletionPanel: FC = observer(() => {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;

  const scrollToBottom = () => {
    if (inputRef.current)
      inputRef.current.scrollTop = inputRef.current.scrollHeight;
  };

  useEffect(() => {
    if (inputRef.current)
      inputRef.current.style.height = '100%';
    scrollToBottom();
  }, []);

  const setPreset = (preset: CompletionPreset) => {
    commonStore.setCompletionPreset({
      ...preset,
      prompt: t(preset.prompt)
    });
  };

  if (!commonStore.completionPreset)
    setPreset(defaultPresets[0]);

  const name = commonStore.completionPreset!.name;

  const prompt = commonStore.completionPreset!.prompt;
  const setPrompt = (prompt: string) => {
    commonStore.setCompletionPreset({
      ...commonStore.completionPreset!,
      prompt
    });
  };

  const params = commonStore.completionPreset!.params;
  const setParams = (newParams: Partial<CompletionParams>) => {
    commonStore.setCompletionPreset({
      ...commonStore.completionPreset!,
      params: {
        ...commonStore.completionPreset!.params,
        ...newParams
      }
    });
  };

  const onSubmit = (prompt: string) => {
    if (commonStore.status.status === ModelStatus.Offline) {
      toast(t('Please click the button in the top right corner to start the model'), { type: 'warning' });
      commonStore.setCompletionGenerating(false);
      return;
    }

    prompt += params.injectStart.replaceAll('\\n', '\n');

    let answer = '';
    completionSseController = new AbortController();
    fetchEventSource(`http://127.0.0.1:${port}/completions`, // https://api.openai.com/v1/completions || http://127.0.0.1:${port}/completions
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer sk-`
        },
        body: JSON.stringify({
          prompt,
          stream: true,
          model: 'text-davinci-003',
          max_tokens: params.maxResponseToken,
          temperature: params.temperature,
          top_p: params.topP,
          presence_penalty: params.presencePenalty,
          frequency_penalty: params.frequencyPenalty,
          stop: params.stop.replaceAll('\\n', '\n') || undefined
        }),
        signal: completionSseController?.signal,
        onmessage(e) {
          console.log('sse message', e);
          scrollToBottom();
          if (e.data === '[DONE]') {
            commonStore.setCompletionGenerating(false);
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
            answer += data.choices[0].text;
            setPrompt(prompt + answer.trim() + params.injectEnd.replaceAll('\\n', '\n'));
          }
        },
        onclose() {
          console.log('Connection closed');
        },
        onerror(err) {
          commonStore.setCompletionGenerating(false);
          throw err;
        }
      });
  };

  return (
    <div className="flex flex-col sm:flex-row gap-2 overflow-hidden grow">
      <Textarea
        ref={inputRef}
        className="grow"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <div className="flex flex-col gap-1 max-h-48 sm:max-w-sm sm:max-h-full">
        <Dropdown style={{ minWidth: 0 }}
          value={t(commonStore.completionPreset!.name)!}
          selectedOptions={[commonStore.completionPreset!.name]}
          onOptionSelect={(_, data) => {
            if (data.optionValue) {
              setPreset(defaultPresets.find((preset) => preset.name === data.optionValue)!);
            }
          }}>
          {
            defaultPresets.map((preset) =>
              <Option key={preset.name} value={preset.name}>{t(preset.name)!}</Option>)
          }
        </Dropdown>
        <div className="flex flex-col gap-1 overflow-x-hidden overflow-y-auto">
          <Labeled flex breakline label={t('Max Response Token')}
            desc={t('By default, the maximum number of tokens that can be answered in a single response, it can be changed by the user by specifying API parameters.')}
            content={
              <ValuedSlider value={params.maxResponseToken} min={100} max={8100}
                step={400}
                input
                onChange={(e, data) => {
                  setParams({
                    maxResponseToken: data.value
                  });
                }} />
            } />
          <Labeled flex breakline label={t('Temperature')}
            desc={t('Sampling temperature, it\'s like giving alcohol to a model, the higher the stronger the randomness and creativity, while the lower, the more focused and deterministic it will be.')}
            content={
              <ValuedSlider value={params.temperature} min={0} max={2} step={0.1}
                input
                onChange={(e, data) => {
                  setParams({
                    temperature: data.value
                  });
                }} />
            } />
          <Labeled flex breakline label={t('Top_P')}
            desc={t('Just like feeding sedatives to the model. Consider the results of the top n% probability mass, 0.1 considers the top 10%, with higher quality but more conservative, 1 considers all results, with lower quality but more diverse.')}
            content={
              <ValuedSlider value={params.topP} min={0} max={1} step={0.1} input
                onChange={(e, data) => {
                  setParams({
                    topP: data.value
                  });
                }} />
            } />
          <Labeled flex breakline label={t('Presence Penalty')}
            desc={t('Positive values penalize new tokens based on whether they appear in the text so far, increasing the model\'s likelihood to talk about new topics.')}
            content={
              <ValuedSlider value={params.presencePenalty} min={-2} max={2}
                step={0.1} input
                onChange={(e, data) => {
                  setParams({
                    presencePenalty: data.value
                  });
                }} />
            } />
          <Labeled flex breakline label={t('Frequency Penalty')}
            desc={t('Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model\'s likelihood to repeat the same line verbatim.')}
            content={
              <ValuedSlider value={params.frequencyPenalty} min={-2} max={2}
                step={0.1} input
                onChange={(e, data) => {
                  setParams({
                    frequencyPenalty: data.value
                  });
                }} />
            } />
          <Labeled flex breakline label={t('Stop Sequences')}
            desc={t('When this content appears in the response result, the generation will end.')}
            content={
              <Input value={params.stop}
                onChange={(e, data) => {
                  setParams({
                    stop: data.value
                  });
                }} />
            } />
          <Labeled flex breakline label={t('Inject start text')}
            desc={t('Before the response starts, inject this content.')}
            content={
              <Input value={params.injectStart}
                onChange={(e, data) => {
                  setParams({
                    injectStart: data.value
                  });
                }} />
            } />
          <Labeled flex breakline label={t('Inject end text')}
            desc={t('When response finished, inject this content.')}
            content={
              <Input value={params.injectEnd}
                onChange={(e, data) => {
                  setParams({
                    injectEnd: data.value
                  });
                }} />
            } />
        </div>
        <div className="grow" />
        <div className="flex justify-between gap-2">
          <Button className="grow" onClick={() => {
            setPreset(defaultPresets.find((preset) => preset.name === name)!);
          }}>{t('Reset')}</Button>
          <Button className="grow" appearance="primary" onClick={() => {
            if (commonStore.completionGenerating) {
              completionSseController?.abort();
              commonStore.setCompletionGenerating(false);
            } else {
              commonStore.setCompletionGenerating(true);
              onSubmit(prompt);
            }
          }}>{!commonStore.completionGenerating ? t('Generate') : t('Stop')}</Button>
        </div>
      </div>
    </div>
  );
});

export const Completion: FC = observer(() => {
  return (
    <div className="flex flex-col gap-1 p-2 h-full overflow-hidden">
      <WorkHeader />
      <CompletionPanel />
    </div>
  );
});
