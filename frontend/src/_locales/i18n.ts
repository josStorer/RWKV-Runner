import i18n, {changeLanguage} from 'i18next';
import {resources} from './resources';
import {getUserLanguage} from '../utils';

i18n.init({
  resources
}).then(() => {
  changeLanguage(getUserLanguage());
});
