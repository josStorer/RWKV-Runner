import { FC, ReactElement, useEffect, useRef, useState } from 'react'
import { Button } from '@fluentui/react-components'
import { ArrowRight20Regular } from '@fluentui/react-icons'
import { observer } from 'mobx-react-lite'
import commonStore from '../stores/commonStore'

export const MobileFloatingNavigator: FC<{
  autoHideDelay?: number
  topTabList: ReactElement
  bottomTabList: ReactElement
}> = observer(({ autoHideDelay = 3000, topTabList, bottomTabList }) => {
  const useDarkMode = commonStore.settings.darkMode
  const [expanded, setExpanded] = useState(true)
  const ref = useRef<HTMLDivElement>(null)
  const timeout = useRef(autoHideDelay)

  useEffect(() => {
    const timer = setInterval(() => {
      if (expanded) {
        timeout.current -= 100
        if (timeout.current <= 0) {
          setExpanded(false)
        }
      }
    }, 100)
    return () => {
      clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    const listener = (e: UIEvent) => {
      if (ref.current) {
        if (ref.current.contains(e.target as Node)) {
          setExpanded(true)
          timeout.current = autoHideDelay
        } else {
          setExpanded(false)
        }
      }
    }
    document.addEventListener('mousedown', listener)
    document.addEventListener('touchstart', listener)
    return () => {
      document.removeEventListener('mousedown', listener)
      document.removeEventListener('touchstart', listener)
    }
  }, [])

  return (
    <div
      ref={ref}
      className={
        'absolute ml-2 flex h-screen w-10 flex-col items-center justify-center'
      }
    >
      <div
        className={`z-[1000] rounded-md border border-black ${useDarkMode ? 'border-opacity-50' : 'border-opacity-30'} ${useDarkMode ? 'bg-black' : 'bg-white'} ${useDarkMode ? 'bg-opacity-10' : 'bg-opacity-30'} backdrop-blur`}
        style={{ transformOrigin: 'top center' }}
      >
        {expanded ? (
          <div className="flex flex-col">
            {topTabList}
            <div className="mr-1 h-px bg-gray-500"></div>
            {bottomTabList}
          </div>
        ) : (
          <Button icon={<ArrowRight20Regular />} appearance="subtle"></Button>
        )}
      </div>
    </div>
  )
})
