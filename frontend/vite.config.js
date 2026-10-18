import { defineConfig } from 'vite';
export default defineConfig({
  base: './',
  build: { outDir: '../hub/static', emptyOutDir: true, assetsDir: 'assets' },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
  test: { environment: 'jsdom', setupFiles: './src/test-setup.js', restoreMocks: true },
});
