// @ts-ignore
import { dependencies } from './package.json';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';

// dependencies that exist anywhere
const vendor = [
  'react', 'react-dom', 'react-router', 'react-router-dom',
  '@fluentui/react-icons',
  'mobx', 'mobx-react-lite',
  'i18next', 'react-i18next',
  'usehooks-ts', 'react-toastify',
  'classnames'
];

const embedded = [
  // split @fluentui/react-components by components
  '@fluentui/react-components',

  // dependencies that exist in single component
  'react-beautiful-dnd',
  '@magenta/music', 'html-midi-player',
  'react-markdown', 'rehype-highlight', 'rehype-raw', 'remark-breaks', 'remark-gfm'
];

function renderChunks(deps: Record<string, string>) {
  let chunks = {};
  Object.keys(deps).forEach((key) => {
    if ([...vendor, ...embedded].includes(key)) return;
    chunks[key] = [key];
  });
  return chunks;
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(),
    visualizer({
      template: 'treemap',
      gzipSize: true,
      brotliSize: true
    })],
  build: {
    chunkSizeWarningLimit: 3000,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor,
          ...renderChunks(dependencies)
        }
      }
    }
  }
});
