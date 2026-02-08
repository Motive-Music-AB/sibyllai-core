import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': {
        // Backend port (8003 to avoid conflict with other apps)
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8003',
        ws: true,
      },
    },
  },
})
