import { useCallback, useState } from 'react'
import { Button, OverlayDrawer } from '@fluentui/react-components'
import classNames from 'classnames'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import commonStore from '../stores/commonStore'

export const BottomLogger = observer(() => {
  const [isOpen, setIsOpen] = useState(false)
  const { t } = useTranslation()
  const isDark = commonStore.settings.darkMode

  const onClickBottomButton = useCallback(() => {
    setIsOpen(true)
  }, [])

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
        {t('Open logger')}
      </Button>
    </div>
  )
})
