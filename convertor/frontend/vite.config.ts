import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

/**
 * Vite Configuration for Premium React TypeScript Application
 * 
 * Features:
 * - React plugin with Fast Refresh for rapid development
 * - Path aliases for clean imports
 * - Optimized production build settings
 * - Proxy configuration for backend API
 */
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url)),
            '@components': fileURLToPath(new URL('./src/components', import.meta.url)),
            '@hooks': fileURLToPath(new URL('./src/hooks', import.meta.url)),
            '@context': fileURLToPath(new URL('./src/context', import.meta.url)),
            '@types': fileURLToPath(new URL('./src/types', import.meta.url)),
            '@api': fileURLToPath(new URL('./src/api', import.meta.url)),
            '@styles': fileURLToPath(new URL('./src/styles', import.meta.url)),
        },
    },
    server: {
        port: 3000,
        proxy: {
            '/api': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false,
            },
        },
    },
    build: {
        // Optimized chunk splitting for better caching
        rollupOptions: {
            output: {
                manualChunks: {
                    'react-vendor': ['react', 'react-dom'],
                    'animation-vendor': ['framer-motion'],
                    'syntax-vendor': ['prismjs', 'katex', 'mermaid'],
                },
            },
        },
        // Enable source maps for production debugging
        sourcemap: true,
        // Minification settings for premium performance
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true,
                drop_debugger: true,
            },
        },
    },
    // Optimize dependencies for faster dev server startup
    optimizeDeps: {
        include: ['react', 'react-dom', 'framer-motion', 'prismjs', 'katex'],
    },
});
