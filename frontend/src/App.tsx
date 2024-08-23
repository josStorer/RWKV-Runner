// reference: https://github.com/oliverschwendener/electron-fluent-ui
//
// MIT License
//
// Copyright (c) 2023 josStorer
// Copyright (c) 2023 oliverschwendener
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import { FC, useEffect, useState } from 'react'
import {
  FluentProvider,
  Tab,
  TabList,
  webDarkTheme,
  webLightTheme,
} from '@fluentui/react-components'
import { observer } from 'mobx-react-lite'
import { useTranslation } from 'react-i18next'
import { Route, Routes, useLocation, useNavigate } from 'react-router'
import { useMediaQuery } from 'usehooks-ts'
import { CustomToastContainer } from './components/CustomToastContainer'
import { LazyImportComponent } from './components/LazyImportComponent'
import { MobileFloatingNavigator } from './components/MobileFloatingNavigator'
import { pages as clientPages } from './pages'
import commonStore from './stores/commonStore'

const App: FC = observer(() => {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const location = useLocation()
  const mq = useMediaQuery('(min-width: 640px)')

  const isWeb = commonStore.platform === 'web'
  const screenWidthSmallerThan640 = !mq
  const useMobileStyle = isWeb && screenWidthSmallerThan640

  const pages = isWeb
    ? clientPages.filter(
        (page) =>
          !['/configs', '/models', '/downloads', '/train', '/about'].some(
            (path) => page.path === path
          )
      )
    : clientPages

  const [path, setPath] = useState<string>(pages[0].path)
  const isHome = path === '/'

  const selectTab = (selectedPath: unknown) =>
    typeof selectedPath === 'string'
      ? navigate({ pathname: selectedPath })
      : null

  useEffect(() => setPath(location.pathname), [location])

  const topTabList = (
    <TabList
      size="large"
      appearance="subtle"
      selectedValue={path}
      onTabSelect={(_, { value }) => selectTab(value)}
      vertical
    >
      {pages
        .filter((page) => page.top)
        .map(({ label, path, icon }, index) => (
          <Tab icon={icon} key={`${path}-${index}`} value={path}>
            {mq && t(label)}
          </Tab>
        ))}
    </TabList>
  )
  const bottomTabList = (
    <TabList
      size="large"
      appearance="subtle"
      selectedValue={path}
      onTabSelect={(_, { value }) => selectTab(value)}
      vertical
    >
      {pages
        .filter((page) => !page.top)
        .map(({ label, path, icon }, index) => (
          <Tab icon={icon} key={`${path}-${index}`} value={path}>
            {mq && t(label)}
          </Tab>
        ))}
    </TabList>
  )

  return (
    <FluentProvider
      theme={commonStore.settings.darkMode ? webDarkTheme : webLightTheme}
      data-theme={commonStore.settings.darkMode ? 'dark' : 'light'}
    >
      <div className="flex h-screen">
        {useMobileStyle ? (
          !isHome ? (
            <MobileFloatingNavigator
              topTabList={topTabList}
              bottomTabList={bottomTabList}
            />
          ) : (
            <></>
          )
        ) : (
          <div className="flex w-16 flex-col justify-between p-2 sm:w-48">
            {topTabList}
            {bottomTabList}
          </div>
        )}
        <div
          className={
            'box-border h-full w-full overflow-y-hidden py-2 pr-2' +
            (useMobileStyle ? ' pl-2' : '')
          }
        >
          <Routes>
            {pages.map(({ path, element }, index) => (
              <Route
                key={`${path}-${index}`}
                path={path}
                element={<LazyImportComponent lazyChildren={element} />}
              />
            ))}
          </Routes>
        </div>
      </div>
      <CustomToastContainer />
    </FluentProvider>
  )
})

export default App
