import { FC, ReactElement, useEffect, useRef, useState } from 'react'
import { Button } from '@fluentui/react-components'
import { ArrowRight20Regular } from '@fluentui/react-icons'
import classNames from 'classnames'
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

  const contentRef = useRef<HTMLDivElement>(null)
  const [height, setHeight] = useState(0)

  useEffect(() => {
    const updateHeight = () => {
      setHeight(contentRef.current?.scrollHeight || 0)
    }
    const observer = new ResizeObserver(updateHeight)
    observer.observe(contentRef.current!)
    return () => observer.disconnect()
  }, [])

  return (
    <div
      ref={ref}
      className={classNames(
        'absolute',
        'flex',
        'flex-col',
        'h-screen',
        'items-center',
        'justify-center',
        'ml-2',
        'w-10'
      )}
    >
      <div
        style={{ height: `${height + 2}px` }}
        className={classNames(
          'backdrop-blur',
          'border',
          'border-black',
          'duration-500',
          'ease-in-out"',
          'overflow-hidden',
          'overflow-hidden',
          'rounded-md',
          'transition-all',
          'z-[10000]',
          useDarkMode ? 'bg-black' : 'bg-white',
          useDarkMode ? 'bg-opacity-10' : 'bg-opacity-30',
          useDarkMode ? 'border-opacity-50' : 'border-opacity-30'
        )}
      >
        <div
          ref={contentRef}
          className={classNames('flex', 'flex-col', 'justify-center')}
        >
          {expanded ? (
            <>
              {topTabList}
              <div className="ml-1 mr-1 h-px bg-gray-500"></div>
              {bottomTabList}
            </>
          ) : (
            <Button icon={<ArrowRight20Regular />} appearance="subtle"></Button>
          )}
        </div>
      </div>
    </div>
  )
})
