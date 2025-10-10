// Execute this script on the Hugging Face files list page to export JSON data. Don't forget to click "Load more files".
// Run console.log(JSON.stringify(modelsJson, null, 2)) to output the JSON to the console.

let modelsJson = []

function extractValue(text, prefix) {
  let ret
  text.split('\n').forEach(line => {
    if (!ret && line.startsWith(prefix))
      ret = line.replace(prefix, '').trim()
  })
  return ret || ''
}

document.querySelectorAll('ul.rounded-b-lg li.grid').forEach(async e => {
  let data = {}
  data.name = e.children[0].children[0].textContent.trim()

  if (!data.name.endsWith('.bin') && !data.name.endsWith('.pth') && !data.name.endsWith('.gguf'))
    return

  data.desc = { en: '', zh: '', ja: '' }
  const rawText = await (await fetch(e.children[1].href.replace('/resolve/', '/raw/'))).text()

  data.size = parseInt(extractValue(rawText, 'size'))
  data.SHA256 = extractValue(rawText, 'oid sha256:')
  data.lastUpdated = e.children[3].children[0].getAttribute('datetime')
  data.url = e.children[1].href.replace('/resolve/', '/blob/').replace('?download=true', '')
  data.downloadUrl = e.children[1].href.replace('?download=true', '')
  data.tags = []

  modelsJson.push(data)
})

setTimeout(() => {
  console.log(JSON.stringify(modelsJson, null, 2))
}, 500)
