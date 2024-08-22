import { FC, useEffect, useRef, useState } from 'react'
import { Tab, TabList } from '@fluentui/react-components'
import { observer } from 'mobx-react-lite'
import { useLocation, useNavigate } from 'react-router'
import { pages as clientPages } from '../pages'
import commonStore from '../stores/commonStore'

export const MobileFloatingNavigator: FC<{
  autoHideDelay?: number
}> = observer(({ autoHideDelay = 3000 }) => {
  const navigate = useNavigate()
  const location = useLocation()

  const useDarkMode = commonStore.settings.darkMode

  const isWeb = commonStore.platform === 'web'
  if (!isWeb) return <></>

  const [expanded, setExpaned] = useState(false)

  const pages = clientPages.filter((page) => {
    return !['/configs', '/models', '/downloads', '/train', '/about'].some(
      (path) => page.path === path
    )
  })

  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!expanded) return
    const timer = setTimeout(() => {
      setExpaned(false)
      if (ref.current) {
        const elements = ref.current.querySelectorAll<HTMLElement>('*')
        elements.forEach((el) => el.blur())
      }
    }, autoHideDelay)
    return () => clearTimeout(timer)
  }, [expanded])

  const [currentPath, setCurrentPath] = useState(pages[0].path)

  useEffect(() => setCurrentPath(location.pathname), [location])

  const selectTab = (selectedPath: unknown) => {
    if (!expanded) {
      setExpaned(true)
      return
    }

    if (typeof selectedPath === 'string') {
      navigate({ pathname: selectedPath })
    }
  }

  return (
    <div
      ref={ref}
      className={
        'pointer-events-none absolute ml-2 flex h-screen w-10 flex-col items-center justify-center'
      }
    >
      <div
        className={`pr pointer-events-auto z-50 flex flex-col rounded-md border border-black ${useDarkMode ? 'border-opacity-50' : 'border-opacity-30'} ${useDarkMode ? 'bg-black' : 'bg-white'} ${useDarkMode ? 'bg-opacity-10' : 'bg-opacity-30'} ${expanded ? 'pl-1' : 'p-0'} backdrop-blur`}
        style={{ transformOrigin: 'top center' }}
      >
        <TabList
          size="large"
          appearance="subtle"
          selectedValue={expanded ? currentPath : null}
          onTabSelect={(_, { value }) => selectTab(value)}
          vertical
        >
          {pages
            .filter((page) => page.top)
            .map(({ path, icon }, index) => {
              if (expanded || (!expanded && currentPath === path)) {
                return (
                  <Tab
                    className={`${expanded ? '' : ''}`}
                    icon={icon}
                    key={`${path}-${index}`}
                    value={path}
                  />
                )
              }
              return null
            })}
        </TabList>
        {expanded && <div className="mr-1 h-px bg-gray-500"></div>}
        <TabList
          size="large"
          appearance="subtle"
          selectedValue={expanded ? currentPath : null}
          onTabSelect={(_, { value }) => selectTab(value)}
          vertical
        >
          {pages
            .filter((page) => !page.top)
            .map(({ path, icon }, index) => {
              if (expanded || (!expanded && currentPath === path)) {
                return (
                  <Tab
                    className={`${expanded ? '' : ''}`}
                    icon={icon}
                    key={`${path}-${index}`}
                    value={path}
                  />
                )
              }
              return null
            })}
        </TabList>
      </div>
    </div>
  )
})
