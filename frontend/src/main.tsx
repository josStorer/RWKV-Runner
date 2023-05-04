import React from 'react'
import {createRoot} from 'react-dom/client'
import './style.css'
import App from './App'
import {HashRouter} from 'react-router-dom';

const container = document.getElementById('root')

const root = createRoot(container!)

root.render(
    <HashRouter>
        <App/>
    </HashRouter>
)
