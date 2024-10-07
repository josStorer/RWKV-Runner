import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, OverlayDrawer } from '@fluentui/react-components'
import classNames from 'classnames'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
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
  console.log('Raw taskMap data:', Array.from(taskMap.entries()))

  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Scroll to the bottom when items array changes
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [taskMap])

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
              // TODO: Why backdrop-blur is not working at there?
              'backdrop-blur',
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
            <div className={classNames('flex', 'justify-end')}>
              {/* TODO: Where is "Close" icon in "fluentui/react-icons"? */}
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
                'flex-col'
              )}
            >
              {Array.from(taskMap.keys()).map((key: string) => {
                const task = taskMap.get(key)
                const lines = task?.lines
                return lines?.map((line, index) => {
                  return (
                    <div
                      key={index}
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
                })
              })}
            </div>

            {/* TODO: render user interaction components to hide logger */}
            {/* TODO: render WSL, which can be found at Train.tsx: commonStore.wslStdout */}
            {/* TODO: What to render? The terminal window opened by this process? */}
            {/* TODO: Handle this warning: "A Dialog should have at least one focusable element inside DialogSurface."" */}
          </div>
        </div>
      </OverlayDrawer>

      {/* TODO: change style */}
      {/* TODO: Watch from backend */}
      <Button appearance="secondary" size="small" onClick={onClickBottomButton}>
        {t('Open Terminal')}
      </Button>
    </div>
  )
})
