import React from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';
import 'react-toastify/dist/ReactToastify.css';
import App from './App';
import {HashRouter} from 'react-router-dom';
import {startup} from './startup';
import './_locales/i18n-react';

startup().then(() => {
  const container = document.getElementById('root');

  const root = createRoot(container!);

  root.render(
    <HashRouter>
      <App/>
    </HashRouter>
  );
});
