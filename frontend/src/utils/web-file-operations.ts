import { getDocument, GlobalWorkerOptions, PDFDocumentProxy } from 'pdfjs-dist'
import { TextItem } from 'pdfjs-dist/types/src/display/api'

export function webOpenOpenFileDialog(
  filterPattern: string,
  fnStartLoading: Function | undefined
): Promise<{
  blob: File
  content?: string
}> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = filterPattern
      .replaceAll('*.txt', 'text/plain')
      .replace('*.midi', 'audio/midi')
      .replace('*.mid', 'audio/midi')
      .replaceAll('*.', 'application/')
      .replaceAll(';', ',')

    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      if (fnStartLoading && typeof fnStartLoading === 'function')
        fnStartLoading()
      if (!GlobalWorkerOptions.workerSrc && file.type === 'application/pdf')
        GlobalWorkerOptions.workerSrc = await import(
          // @ts-ignore
          'pdfjs-dist/build/pdf.worker.min.mjs'
        )
      if (file.type === 'text/plain') {
        const reader = new FileReader()
        reader.readAsText(file, 'UTF-8')

        reader.onload = (event) => {
          const content = event.target?.result as string
          resolve({
            blob: file,
            content: content,
          })
        }
        reader.onerror = reject
      } else if (file.type === 'application/pdf') {
        const readPDFPage = async (doc: PDFDocumentProxy, pageNo: number) => {
          const page = await doc.getPage(pageNo)
          const tokenizedText = await page.getTextContent()
          return tokenizedText.items
            .map((token) => (token as TextItem).str)
            .join('')
        }
        let reader = new FileReader()
        reader.readAsArrayBuffer(file)

        reader.onload = async (event) => {
          try {
            const doc = await getDocument(event.target?.result!).promise
            const pageTextPromises = []
            for (let pageNo = 1; pageNo <= doc.numPages; pageNo++) {
              pageTextPromises.push(readPDFPage(doc, pageNo))
            }
            const pageTexts = await Promise.all(pageTextPromises)
            let content
            if (pageTexts.length === 1) content = pageTexts[0]
            else
              content = pageTexts
                .map((p, i) => `Page ${i + 1}:\n${p}`)
                .join('\n\n')
            resolve({
              blob: file,
              content: content,
            })
          } catch (err) {
            reject(err)
          }
        }
        reader.onerror = reject
      } else {
        resolve({
          blob: file,
        })
      }
    }
    input.click()
  })
}
