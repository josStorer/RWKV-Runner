import React, { FC, useEffect, useRef } from 'react'
import {
  Button,
  Dropdown,
  Input,
  Option,
  Textarea,
} from '@fluentui/react-components'
import { ArrowSync20Regular } from '@fluentui/react-icons'
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { toast } from 'react-toastify'
import { DialogButton } from '../components/DialogButton'
import { Labeled } from '../components/Labeled'
import { ToolTipButton } from '../components/ToolTipButton'
import { ValuedSlider } from '../components/ValuedSlider'
import { WorkHeader } from '../components/WorkHeader'
import commonStore, { ModelStatus } from '../stores/commonStore'
import { CompletionParams, CompletionPreset } from '../types/completion'
import { getReqUrl } from '../utils'
import { defaultPenaltyDecay, defaultPresets } from './defaultConfigs'
import { PresetsButton } from './PresetsManager/PresetsButton'

let completionSseController: AbortController | null = null

const CompletionPanel: FC = observer(() => {
  const { t } = useTranslation()
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const port = commonStore.getCurrentModelConfig().apiParameters.apiPort

  const scrollToBottom = () => {
    if (inputRef.current)
      inputRef.current.scrollTop = inputRef.current.scrollHeight
  }

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = '100%'
      inputRef.current.style.maxHeight = '100%'
    }
    scrollToBottom()
  }, [])

  const setPreset = (preset: CompletionPreset) => {
    commonStore.setCompletionSubmittedPrompt(t(preset.prompt))
    commonStore.setCompletionPreset({
      ...preset,
      prompt: t(preset.prompt),
    })
  }

  if (!commonStore.completionPreset) setPreset(defaultPresets[0])

  const name = commonStore.completionPreset!.name

  const prompt = commonStore.completionPreset!.prompt
  const setPrompt = (prompt: string) => {
    commonStore.setCompletionPreset({
      ...commonStore.completionPreset!,
      prompt,
    })
  }

  const params = commonStore.completionPreset!.params
  const setParams = (newParams: Partial<CompletionParams>) => {
    commonStore.setCompletionPreset({
      ...commonStore.completionPreset!,
      params: {
        ...commonStore.completionPreset!.params,
        ...newParams,
      },
    })
  }

  const onSubmit = async (prompt: string) => {
    commonStore.setCompletionSubmittedPrompt(prompt)

    if (
      commonStore.status.status === ModelStatus.Offline &&
      !commonStore.settings.apiUrl &&
      commonStore.platform !== 'web'
    ) {
      toast(
        t('Please click the button in the top right corner to start the model'),
        { type: 'warning' }
      )
      commonStore.setCompletionGenerating(false)
      return
    }

    prompt += params.injectStart.replaceAll('\\n', '\n')

    let answer = ''
    let finished = false
    const finish = () => {
      finished = true
      commonStore.setCompletionGenerating(false)
    }
    completionSseController = new AbortController()
    const { url, headers } = await getReqUrl(port, '/v1/completions', true)
    fetchEventSource(
      // https://api.openai.com/v1/completions || http://127.0.0.1:${port}/v1/completions
      url,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${commonStore.settings.apiKey}`,
          ...headers,
        },
        body: JSON.stringify({
          prompt,
          stream: true,
          model: commonStore.settings.apiCompletionModelName, // 'text-davinci-003'
          max_tokens: params.maxResponseToken,
          temperature: params.temperature,
          top_p: params.topP,
          presence_penalty: params.presencePenalty,
          frequency_penalty: params.frequencyPenalty,
          stop: params.stop.replaceAll('\\n', '\n') || undefined,
          penalty_decay:
            !params.penaltyDecay || params.penaltyDecay === defaultPenaltyDecay
              ? undefined
              : params.penaltyDecay,
        }),
        signal: completionSseController?.signal,
        onmessage(e) {
          if (finished) return
          scrollToBottom()
          if (e.data.trim() === '[DONE]') {
            finish()
            return
          }
          let data
          try {
            data = JSON.parse(e.data)
          } catch (error) {
            console.debug('json error', error)
            return
          }
          if (data.model) commonStore.setLastModelName(data.model)
          if (
            data.choices &&
            Array.isArray(data.choices) &&
            data.choices.length > 0
          ) {
            answer +=
              data.choices[0]?.text || data.choices[0]?.delta?.content || ''
            setPrompt(
              prompt +
                answer.replace(/\s+$/, '') +
                params.injectEnd.replaceAll('\\n', '\n')
            )

            if (data.choices[0]?.finish_reason) {
              finish()
              return
            }
          }
        },
        async onopen(response) {
          if (response.status !== 200) {
            toast(
              response.status +
                ' - ' +
                response.statusText +
                ' - ' +
                (await response.text()),
              {
                type: 'error',
              }
            )
          }
        },
        onclose() {
          console.log('Connection closed')
        },
        onerror(err) {
          err = err.message || err
          if (err && !err.includes('ReadableStreamDefaultReader'))
            toast(err, {
              type: 'error',
            })
          commonStore.setCompletionGenerating(false)
          throw err
        },
      }
    )
  }

  return (
    <div className="flex grow flex-col gap-2 overflow-hidden sm:flex-row">
      <Textarea
        ref={inputRef}
        className="grow"
        value={prompt}
        onChange={(e) => {
          commonStore.setCompletionSubmittedPrompt(e.target.value)
          setPrompt(e.target.value)
        }}
      />
      <div className="flex max-h-48 flex-col gap-1 sm:max-h-full sm:max-w-sm">
        <div className="flex gap-2">
          <Dropdown
            style={{ minWidth: 0 }}
            className="grow"
            value={t(commonStore.completionPreset!.name)!}
            selectedOptions={[commonStore.completionPreset!.name]}
            onOptionSelect={(_, data) => {
              if (data.optionValue) {
                setPreset(
                  defaultPresets.find(
                    (preset) => preset.name === data.optionValue
                  )!
                )
              }
            }}
          >
            {defaultPresets.map((preset) => (
              <Option key={preset.name} value={preset.name}>
                {t(preset.name)!}
              </Option>
            ))}
          </Dropdown>
          <PresetsButton tab="Completion" />
        </div>
        <div className="flex flex-col gap-1 overflow-y-auto overflow-x-hidden p-1">
          <Labeled
            flex
            breakline
            label={t('Max Response Token')}
            desc={t(
              'By default, the maximum number of tokens that can be answered in a single response, it can be changed by the user by specifying API parameters.'
            )}
            content={
              <ValuedSlider
                value={params.maxResponseToken}
                min={100}
                max={8100}
                step={100}
                input
                onChange={(e, data) => {
                  setParams({
                    maxResponseToken: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Temperature')}
            desc={t(
              "Sampling temperature, it's like giving alcohol to a model, the higher the stronger the randomness and creativity, while the lower, the more focused and deterministic it will be."
            )}
            content={
              <ValuedSlider
                value={params.temperature}
                min={0}
                max={3}
                step={0.1}
                input
                onChange={(e, data) => {
                  setParams({
                    temperature: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Top_P')}
            desc={t(
              'Just like feeding sedatives to the model. Consider the results of the top n% probability mass, 0.1 considers the top 10%, with higher quality but more conservative, 1 considers all results, with lower quality but more diverse.'
            )}
            content={
              <ValuedSlider
                value={params.topP}
                min={0}
                max={1}
                step={0.05}
                input
                onChange={(e, data) => {
                  setParams({
                    topP: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Presence Penalty')}
            desc={t(
              "Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."
            )}
            content={
              <ValuedSlider
                value={params.presencePenalty}
                min={0}
                max={2}
                step={0.1}
                input
                onChange={(e, data) => {
                  setParams({
                    presencePenalty: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Frequency Penalty')}
            desc={t(
              "Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."
            )}
            content={
              <ValuedSlider
                value={params.frequencyPenalty}
                min={0}
                max={2}
                step={0.1}
                input
                onChange={(e, data) => {
                  setParams({
                    frequencyPenalty: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={
              t('Penalty Decay') +
              (!params.penaltyDecay ||
              params.penaltyDecay === defaultPenaltyDecay
                ? ` (${t('Default')})`
                : '')
            }
            desc={t("If you don't know what it is, keep it default.")}
            content={
              <ValuedSlider
                value={params.penaltyDecay || defaultPenaltyDecay}
                min={0.99}
                max={0.999}
                step={0.001}
                toFixed={3}
                input
                onChange={(e, data) => {
                  setParams({
                    penaltyDecay: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Stop Sequences')}
            desc={t(
              'When this content appears in the response result, the generation will end.'
            )}
            content={
              <Input
                value={params.stop}
                onChange={(e, data) => {
                  setParams({
                    stop: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Inject start text')}
            desc={t('Before the response starts, inject this content.')}
            content={
              <Input
                value={params.injectStart}
                onChange={(e, data) => {
                  setParams({
                    injectStart: data.value,
                  })
                }}
              />
            }
          />
          <Labeled
            flex
            breakline
            label={t('Inject end text')}
            desc={t('When response finished, inject this content.')}
            content={
              <Input
                value={params.injectEnd}
                onChange={(e, data) => {
                  setParams({
                    injectEnd: data.value,
                  })
                }}
              />
            }
          />
        </div>
        <div className="grow" />
        <div className="hidden justify-between gap-2 sm:flex">
          <Button
            className="grow"
            onClick={() => {
              const newPrompt = prompt
                .replace(/\n+\ /g, '\n')
                .split('\n')
                .map((line) => line.trim())
                .join('\n')
              setPrompt(newPrompt)
              commonStore.setCompletionSubmittedPrompt(newPrompt)
            }}
          >
            {t('Format Content')}
          </Button>
        </div>
        <div className="flex justify-between gap-2">
          <ToolTipButton
            desc={t('Regenerate')}
            icon={<ArrowSync20Regular />}
            onClick={() => {
              completionSseController?.abort()
              commonStore.setCompletionGenerating(true)
              setPrompt(commonStore.completionSubmittedPrompt)
              onSubmit(commonStore.completionSubmittedPrompt)
            }}
          />
          <DialogButton
            className="grow"
            text={t('Reset')}
            title={t('Reset')}
            content={t(
              'Are you sure you want to reset this page? It cannot be undone.'
            )}
            onConfirm={() => {
              setPreset(defaultPresets.find((preset) => preset.name === name)!)
            }}
          />
          <Button
            className="grow"
            appearance="primary"
            onClick={() => {
              if (commonStore.completionGenerating) {
                completionSseController?.abort()
                commonStore.setCompletionGenerating(false)
              } else {
                commonStore.setCompletionGenerating(true)
                onSubmit(prompt)
              }
            }}
          >
            {!commonStore.completionGenerating ? t('Generate') : t('Stop')}
          </Button>
        </div>
      </div>
    </div>
  )
})

const Completion: FC = observer(() => {
  return (
    <div className="flex h-full flex-col gap-1 overflow-hidden p-2">
      <WorkHeader />
      <CompletionPanel />
    </div>
  )
})

export default Completion
