export type ModelSourceItem = {
  name: string
  desc?: { [lang: string]: string | undefined }
  size: number
  SHA256?: string
  lastUpdated: string
  url?: string
  downloadUrl?: string
  tags?: string[]
  customTokenizer?: string
  hide?: boolean
  functionCall?: boolean

  lastUpdatedMs?: number
  isComplete?: boolean
  isLocal?: boolean
  localSize?: number
}
