import i18n, {changeLanguage} from 'i18next';
import {resources} from './resources';
import {getNavigatorLanguage} from '../utils';

i18n.init({
  resources
}).then(() => {
  changeLanguage(getNavigatorLanguage());
});
