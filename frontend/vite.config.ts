import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Proxy all API routes to backend
      // This avoids CORS issues during development
      '/nodes': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/search': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ask': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Disable buffering for SSE streaming
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // For /ask/stream, ensure we don't buffer
            if (req.url?.includes('/ask/stream')) {
              proxyReq.setHeader('Connection', 'keep-alive');
            }
          });
          proxy.on('proxyRes', (proxyRes, req, res) => {
            // For SSE endpoints, disable buffering
            if (req.url?.includes('/ask/stream')) {
              proxyRes.headers['x-accel-buffering'] = 'no';
            }
          });
        },
      },
      '/events': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/lineage': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/edges': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/relationships': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/_admin': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/admin': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/prometheus': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/openapi.json': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/debug': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
