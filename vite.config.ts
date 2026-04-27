import { defineConfig } from 'vite';

export default defineConfig(({ command }) => ({
  base: command === 'build' ? (process.env.VITE_BASE ?? '/deepvariant-tfjs/') : '/',
  server: { port: 5173, host: '0.0.0.0' },
  build: { target: 'es2022' },
}));
