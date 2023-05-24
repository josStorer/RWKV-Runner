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

export type CompletionParams = Omit<ApiParameters, 'apiPort'> & { stop: string }

export type CompletionPreset = {
  name: string,
  prompt: string,
  params: CompletionParams
}

export const defaultPresets: CompletionPreset[] = [{
  name: 'Writer',
  prompt: 'The following is an epic science fiction masterpiece that is immortalized, with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n',
  params: {
    maxResponseToken: 4100,
    temperature: 1,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: ''
  }
}, {
  name: 'Translator',
  prompt: '',
  params: {
    maxResponseToken: 4100,
    temperature: 1,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: ''
  }
}, {
  name: 'Catgirl',
  prompt: '',
  params: {
    maxResponseToken: 4100,
    temperature: 1,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: ''
  }
}, {
  name: 'Explain Code',
  prompt: '',
  params: {
    maxResponseToken: 4100,
    temperature: 1,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: ''
  }
}, {
  name: 'Werewolf',
  prompt: '',
  params: {
    maxResponseToken: 4100,
    temperature: 1,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: ''
  }
}, {
  name: 'Blank',
  prompt: '',
  params: {
    maxResponseToken: 4100,
    temperature: 1,
    topP: 0.5,
    presencePenalty: 0.4,
    frequencyPenalty: 0.4,
    stop: ''
  }
}];

const CompletionPanel: FC = observer(() => {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort;
  const sseControllerRef = useRef<AbortController | null>(null);

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

    let answer = '';
    sseControllerRef.current = new AbortController();
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
          stop: params.stop || undefined
        }),
        signal: sseControllerRef.current?.signal,
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
            setPrompt(prompt + answer);
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
            desc={t('Sampling temperature, the higher the stronger the randomness and creativity, while the lower, the more focused and deterministic it will be.')}
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
            desc={t('Consider the results of the top n% probability mass, 0.1 considers the top 10%, with higher quality but more conservative, 1 considers all results, with lower quality but more diverse.')}
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
        </div>
        <div className="grow" />
        <div className="flex justify-between gap-2">
          <Button className="grow" onClick={() => {
            setPreset(defaultPresets.find((preset) => preset.name === name)!);
          }}>{t('Reset')}</Button>
          <Button className="grow" appearance="primary" onClick={() => {
            if (commonStore.completionGenerating) {
              sseControllerRef.current?.abort();
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
