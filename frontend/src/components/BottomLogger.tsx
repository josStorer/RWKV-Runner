import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, OverlayDrawer } from '@fluentui/react-components'
import classNames from 'classnames'
import { toJS } from 'mobx'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { toast } from 'react-toastify'
import { GetCmds, KillCmd } from '../../wailsjs/go/backend_golang/App'
import cmdStore from '../stores/cmd'
import commonStore from '../stores/commonStore'

export const BottomLogger = observer(() => {
  const [isOpen, setIsOpen] = useState(false)
  const { t } = useTranslation()
  const isDark = commonStore.settings.darkMode

  const onClickBottomButton = useCallback(() => {
    setIsOpen(true)
  }, [])

  const taskMap = cmdStore.taskMap

  const scrollRef = useRef<HTMLDivElement>(null)

  const rawMap = toJS(taskMap)
  const firstKey = rawMap.keys().next().value
  const currentTask = rawMap.get(firstKey)
  const taskName = currentTask?.name
  const timestamp = currentTask?.timestamp
  const lines = currentTask?.lines ?? []

  const scrollToBottom = () => {
    const current = scrollRef.current
    if (current) {
      current.scrollTo({
        top: current.scrollHeight,
        behavior: 'smooth',
      })
    }
  }

  const clearLines = () => {
    cmdStore.clearLines()
  }

  const copyLines = () => {
    const text = lines.join('\n')
    navigator.clipboard.writeText(text)
    toast(t('Copied to clipboard'), { type: 'success' })
  }

  const endCurrentTask = async () => {
    const cmds = await GetCmds()
    const firstKey = Object.keys(cmds)[0]
    await KillCmd(firstKey)
    cmdStore.clearLines()
  }

  useEffect(() => {
    scrollToBottom()
  }, [lines])

  return (
    <div className={classNames('flex', 'justify-center', 'pb-2')}>
      <OverlayDrawer
        className={classNames()}
        style={{ backgroundColor: 'transparent', boxShadow: 'none' }}
        position={'bottom'}
        open={isOpen}
        onOpenChange={(_, { open }) => setIsOpen(open)}
      >
        <div className={classNames('flex', 'h-full', 'w-full')}>
          <div
            className={classNames(
              'flex-1',
              'm-3',
              'rounded-xl',
              'border-stone-500',
              'border',
              'p-2',
              'w-full',
              'flex',
              'flex-col',
              isDark ? 'bg-black' : 'bg-white'
            )}
          >
            <div
              className={classNames('flex', 'justify-start', 'items-center')}
            >
              {taskName && (
                <>
                  <div className={classNames('mr-2')}>{'Task Name:'}</div>
                  <div className={classNames('')}>{taskName}</div>
                  <div className={classNames('ml-4 mr-2')}>{'Task ID:'}</div>
                  <div className={classNames('')}>{timestamp}</div>
                  <div className={classNames('w-2')} />
                  <Button onClick={endCurrentTask}>{t('End Task')}</Button>
                </>
              )}
              <div className={classNames('flex-1')} />
              {lines.length > 0 && (
                <>
                  <Button onClick={copyLines}>{t('Copy')}</Button>
                  <div className={classNames('w-2')} />
                  <Button onClick={clearLines}>{t('Clear')}</Button>
                  <div className={classNames('w-2')} />
                </>
              )}
              <Button
                onClick={() => {
                  setIsOpen(false)
                }}
              >
                {t('close')}
              </Button>
            </div>

            <div
              ref={scrollRef}
              className={classNames(
                'overflow-y-auto',
                'flex',
                'flex-nowrap',
                'flex-col',
                'mt-1'
              )}
            >
              {lines.map((line, index) => {
                return (
                  <div
                    key={index + line}
                    className={classNames(
                      'w-full',
                      'font-mono',
                      'whitespace-pre-wrap',
                      'text-[8px]'
                    )}
                    style={{ overflowWrap: 'anywhere' }}
                  >
                    {line}
                  </div>
                )
              })}
            </div>

            {/* TODO: render user interaction components to hide logger */}
            {/* TODO: render WSL, which can be found at Train.tsx: commonStore.wslStdout */}
            {/* TODO: What to render? The terminal window opened by this process? */}
            {/* TODO: Handle this warning: "A Dialog should have at least one focusable element inside DialogSurface."" */}
          </div>
        </div>
      </OverlayDrawer>

      {lines.length > 0 && (
        <Button
          appearance="secondary"
          size="small"
          onClick={onClickBottomButton}
        >
          {t('Open Console')}
        </Button>
      )}
    </div>
  )
})
