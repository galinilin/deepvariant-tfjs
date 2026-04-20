import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// In dev, serve from /. In build, prefix with the GH Pages project path so
// asset URLs and import.meta.env.BASE_URL resolve correctly under
// https://<user>.github.io/<repo>/. Override per-build with VITE_BASE.
export default defineConfig(({ command }) => ({
  plugins: [vue()],
  base: command === 'build' ? (process.env.VITE_BASE ?? '/deepvariant-tfjs/') : '/',
  server: { port: 5173, host: '0.0.0.0' },
  build: { target: 'es2022' },
}));
