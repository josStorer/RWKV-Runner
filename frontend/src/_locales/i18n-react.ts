import i18n, { changeLanguage } from 'i18next'
import { initReactI18next } from 'react-i18next'
import { getUserLanguage } from '../utils'
import { resources } from './resources'

i18n
  .use(initReactI18next)
  .init({
    resources,
    interpolation: {
      escapeValue: false, // not needed for react as it escapes by default
    },
  })
  .then(() => {
    changeLanguage(getUserLanguage())
  })
