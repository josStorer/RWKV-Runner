import i18n, { changeLanguage } from 'i18next'
import { getUserLanguage } from '../utils'
import { resources } from './resources'

i18n
  .init({
    resources,
  })
  .then(() => {
    changeLanguage(getUserLanguage())
  })
