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
  defineRuntime('EventsOnMultiple', () => {
  })
  defineRuntime('WindowSetLightTheme', () => {
  })
  defineRuntime('WindowSetDarkTheme', () => {
  })
  defineRuntime('WindowShow', () => {
  })
  defineRuntime('WindowHide', () => {
  })

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
  defineApp('AddToDownloadList', async () => {
  })
  defineApp('ContinueDownload', async () => {
  })
  defineApp('ConvertData', async () => {
  })
  defineApp('ConvertModel', async () => {
  })
  defineApp('ConvertSafetensors', async () => {
  })
  defineApp('CopyFile', async () => {
  })
  defineApp('DeleteFile', async () => {
  })
  defineApp('DepCheck', async () => {
  })
  defineApp('DownloadFile', async () => {
  })
  defineApp('GetPyError', async () => {
  })
  defineApp('InstallPyDep', async () => {
  })
  defineApp('IsPortAvailable', async () => {
  })
  defineApp('MergeLora', async () => {
  })
  defineApp('OpenFileFolder', async () => {
  })
  defineApp('PauseDownload', async () => {
  })
  defineApp('ReadFileInfo', async () => {
  })
  defineApp('RestartApp', async () => {
  })
  defineApp('StartServer', async () => {
  })
  defineApp('StartWebGPUServer', async () => {
  })
  defineApp('UpdateApp', async () => {
  })
  defineApp('WslCommand', async () => {
  })
  defineApp('WslEnable', async () => {
  })
  defineApp('WslInstallUbuntu', async () => {
  })
  defineApp('WslIsEnabled', async () => {
  })
  defineApp('WslStart', async () => {
  })
  defineApp('WslStop', async () => {
  })

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
  defineApp('OpenOpenFileDialog', async (filterPattern) => {
    return new Promise((resolve, reject) => {
      const input = document.createElement('input')
      input.type = 'file'
      input.accept = filterPattern
        .replaceAll('*.txt', 'text/plain')
        .replaceAll('*.', 'application/')
        .replaceAll(';', ',')

      input.onchange = e => {
        const file = e.target?.files[0]
        if (file.type === 'text/plain') {
          const reader = new FileReader()
          reader.readAsText(file, 'UTF-8')

          reader.onload = readerEvent => {
            const content = readerEvent.target?.result
            resolve({
              blob: file,
              content: content
            })
          }
        } else {
          resolve({
            blob: file
          })
        }
      }
      input.click()
    })
  })
  defineApp('OpenSaveFileDialog', async (filterPattern, defaultFileName, savedContent) => {
    const saver = await import('file-saver')
    saver.saveAs(new Blob([savedContent], { type: 'text/plain;charset=utf-8' }), defaultFileName)
    return ''
  })
  defineApp('OpenSaveFileDialogBytes', async (filterPattern, defaultFileName, savedContent) => {
    const saver = await import('file-saver')
    saver.saveAs(new Blob([new Uint8Array(savedContent)], { type: 'octet/stream' }), defaultFileName)
    return ''
  })
  defineApp('ReadJson', async (fileName) => {
    return JSON.parse(localStorage.getItem(fileName))
  })
  defineApp('SaveJson', async (fileName, data) => {
    localStorage.setItem(fileName, JSON.stringify(data))
  })
}