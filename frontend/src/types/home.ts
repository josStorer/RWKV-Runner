import { ReactElement } from 'react'

export type IntroductionContent = {
  [lang: string]: string
}
export type NavCard = {
  label: string
  desc: string
  path: string
  icon: ReactElement
}
