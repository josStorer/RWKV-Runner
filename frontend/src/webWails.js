import { webOpenOpenFileDialog } from './utils/web-file-operations'

function defineRuntime(name, func) {
  window.runtime[name] = func
}

function defineApp(name, func) {
  window.go['backend_golang']['App'][name] = func
}

if (!window.runtime) {
  window.runtime = {}
  document.title += ' WebUI'

  // not implemented
  defineRuntime('EventsOnMultiple', () => {})
  defineRuntime('WindowSetLightTheme', () => {})
  defineRuntime('WindowSetDarkTheme', () => {})
  defineRuntime('WindowShow', () => {})
  defineRuntime('WindowHide', () => {})

  // implemented
  defineRuntime('ClipboardGetText', async () => {
    return await navigator.clipboard.readText()
  })
  defineRuntime('ClipboardSetText', async (text) => {
    await navigator.clipboard.writeText(text)
    return true
  })
  defineRuntime('WindowSetTitle', (title) => {
    document.title = title
  })
  defineRuntime('BrowserOpenURL', (url) => {
    window.open(url, '_blank', 'noopener, noreferrer')
  })
}

if (!window.go) {
  window.go = {}
  window.go['backend_golang'] = {}
  window.go['backend_golang']['App'] = {}

  // not implemented
  defineApp('AddToDownloadList', async () => {})
  defineApp('CloseMidiPort', async () => {})
  defineApp('ContinueDownload', async () => {})
  defineApp('ConvertData', async () => {})
  defineApp('ConvertModel', async () => {})
  defineApp('ConvertSafetensors', async () => {})
  defineApp('CopyFile', async () => {})
  defineApp('DeleteFile', async () => {})
  defineApp('DepCheck', async () => {})
  defineApp('DownloadFile', async () => {})
  defineApp('GetPyError', async () => {})
  defineApp('InstallPyDep', async () => {})
  defineApp('IsPortAvailable', async () => {})
  defineApp('MergeLora', async () => {})
  defineApp('OpenFileFolder', async () => {})
  defineApp('OpenMidiPort', async () => {})
  defineApp('PauseDownload', async () => {})
  defineApp('PlayNote', async () => {})
  defineApp('ReadFileInfo', async () => {})
  defineApp('RestartApp', async () => {})
  defineApp('StartServer', async () => {})
  defineApp('StartWebGPUServer', async () => {})
  defineApp('UpdateApp', async () => {})
  defineApp('WslCommand', async () => {})
  defineApp('WslEnable', async () => {})
  defineApp('WslInstallUbuntu', async () => {})
  defineApp('WslIsEnabled', async () => {})
  defineApp('WslStart', async () => {})
  defineApp('WslStop', async () => {})

  // implemented
  defineApp('FileExists', async () => {
    return false
  })
  defineApp('GetPlatform', async () => {
    return 'web'
  })
  defineApp('ListDirFiles', async () => {
    return []
  })
  defineApp('GetAbsPath', async (path) => {
    return path
  })
  defineApp('GetProxyPort', async () => {
    return 0
  })
  defineApp('OpenOpenFileDialog', webOpenOpenFileDialog)
  defineApp(
    'OpenSaveFileDialog',
    async (filterPattern, defaultFileName, savedContent) => {
      const saver = await import('file-saver')
      saver.saveAs(
        new Blob([savedContent], { type: 'text/plain;charset=utf-8' }),
        defaultFileName
      )
      return ''
    }
  )
  defineApp(
    'OpenSaveFileDialogBytes',
    async (filterPattern, defaultFileName, savedContent) => {
      const saver = await import('file-saver')
      saver.saveAs(
        new Blob([new Uint8Array(savedContent)], { type: 'octet/stream' }),
        defaultFileName
      )
      return ''
    }
  )
  defineApp('ReadJson', async (fileName) => {
    const data = JSON.parse(localStorage.getItem(fileName))
    if (data) return data
    else throw new Error('File not found')
  })
  defineApp('SaveJson', async (fileName, data) => {
    localStorage.setItem(fileName, JSON.stringify(data))
  })
}
