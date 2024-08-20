export type DownloadStatus = {
  name: string
  path: string
  url: string
  transferred: number
  size: number
  speed: number
  progress: number
  downloading: boolean
  done: boolean
}
