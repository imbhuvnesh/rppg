import { defineConfig } from 'vite';
export default defineConfig({
  root: 'demo',
  server: { port: 5173 },
  resolve: { alias: { '@core': '/src/core', '@browser': '/src/browser' } }
});
