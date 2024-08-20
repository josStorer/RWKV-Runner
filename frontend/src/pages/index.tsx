import { FC, lazy, LazyExoticComponent, ReactElement } from 'react'
import {
  ArrowDownload20Regular,
  Chat20Regular,
  ClipboardEdit20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  Home20Regular,
  Info20Regular,
  MusicNote220Regular,
  Settings20Regular,
  Storage20Regular,
} from '@fluentui/react-icons'

type NavigationItem = {
  label: string
  path: string
  icon: ReactElement
  element: LazyExoticComponent<FC>
  top: boolean
}

export const pages: NavigationItem[] = [
  {
    label: 'Home',
    path: '/',
    icon: <Home20Regular />,
    element: lazy(() => import('./Home')),
    top: true,
  },
  {
    label: 'Chat',
    path: '/chat',
    icon: <Chat20Regular />,
    element: lazy(() => import('./Chat')),
    top: true,
  },
  {
    label: 'Completion',
    path: '/completion',
    icon: <ClipboardEdit20Regular />,
    element: lazy(() => import('./Completion')),
    top: true,
  },
  {
    label: 'Composition',
    path: '/composition',
    icon: <MusicNote220Regular />,
    element: lazy(() => import('./Composition')),
    top: true,
  },
  {
    label: 'Configs',
    path: '/configs',
    icon: <DocumentSettings20Regular />,
    element: lazy(() => import('./Configs')),
    top: true,
  },
  {
    label: 'Models',
    path: '/models',
    icon: <DataUsageSettings20Regular />,
    element: lazy(() => import('./Models')),
    top: true,
  },
  {
    label: 'Downloads',
    path: '/downloads',
    icon: <ArrowDownload20Regular />,
    element: lazy(() => import('./Downloads')),
    top: true,
  },
  {
    label: 'Train',
    path: '/train',
    icon: <Storage20Regular />,
    element: lazy(() => import('./Train')),
    top: true,
  },
  {
    label: 'Settings',
    path: '/settings',
    icon: <Settings20Regular />,
    element: lazy(() => import('./Settings')),
    top: false,
  },
  {
    label: 'About',
    path: '/about',
    icon: <Info20Regular />,
    element: lazy(() => import('./About')),
    top: false,
  },
]
