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

import { FluentProvider, Tab, TabList, webDarkTheme, webLightTheme } from '@fluentui/react-components';
import { FC, useEffect, useState } from 'react';
import { Route, Routes, useLocation, useNavigate } from 'react-router';
import { pages as clientPages } from './pages';
import { useMediaQuery } from 'usehooks-ts';
import commonStore from './stores/commonStore';
import { observer } from 'mobx-react-lite';
import { useTranslation } from 'react-i18next';
import { CustomToastContainer } from './components/CustomToastContainer';
import { LazyImportComponent } from './components/LazyImportComponent';

const App: FC = observer(() => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const mq = useMediaQuery('(min-width: 640px)');
  const pages = commonStore.platform === 'web' ? clientPages.filter(page =>
    !['/configs', '/models', '/downloads', '/train', '/about'].some(path => page.path === path)
  ) : clientPages;

  const [path, setPath] = useState<string>(pages[0].path);

  const selectTab = (selectedPath: unknown) =>
    typeof selectedPath === 'string' ? navigate({ pathname: selectedPath }) : null;

  useEffect(() => setPath(location.pathname), [location]);

  return (
    <FluentProvider className="h-screen"
      theme={commonStore.settings.darkMode ? webDarkTheme : webLightTheme}
      data-theme={commonStore.settings.darkMode ? 'dark' : 'light'}>
      <div className="flex h-full">
        <div className="flex flex-col w-16 sm:w-48 p-2 justify-between">
          <TabList
            size="large"
            appearance="subtle"
            selectedValue={path}
            onTabSelect={(_, { value }) => selectTab(value)}
            vertical
          >
            {pages.filter(page => page.top).map(({ label, path, icon }, index) => (
              <Tab icon={icon} key={`${path}-${index}`} value={path}>
                {mq && t(label)}
              </Tab>
            ))}
          </TabList>
          <TabList
            size="large"
            appearance="subtle"
            selectedValue={path}
            onTabSelect={(_, { value }) => selectTab(value)}
            vertical
          >
            {pages.filter(page => !page.top).map(({ label, path, icon }, index) => (
              <Tab icon={icon} key={`${path}-${index}`} value={path}>
                {mq && t(label)}
              </Tab>
            ))}
          </TabList>
        </div>
        <div className="h-full w-full p-2 box-border overflow-y-hidden">
          <Routes>
            {pages.map(({ path, element }, index) => (
              <Route key={`${path}-${index}`} path={path} element={<LazyImportComponent lazyChildren={element} />} />
            ))}
          </Routes>
        </div>
      </div>
      <CustomToastContainer />
    </FluentProvider>
  );
});

export default App;
