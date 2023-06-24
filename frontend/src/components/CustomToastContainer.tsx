import commonStore from '../stores/commonStore';
import { ToastContainer } from 'react-toastify';

export const CustomToastContainer = () =>
  <ToastContainer
    style={{ width: '350px' }}
    position="top-center"
    autoClose={4000}
    pauseOnHover={true}
    hideProgressBar={true}
    newestOnTop={true}
    closeOnClick={false}
    rtl={false}
    pauseOnFocusLoss={false}
    draggable={false}
    theme={commonStore.settings.darkMode ? 'dark' : 'light'}
  />;