/**
 * Application Entry Point
 * 
 * React 18 entry with hash-based routing.
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';

// Import styles
import '@/styles/index.css';
import '@/styles/components.css';
import '@/styles/mainpage.css';
import '@/styles/documentspage.css';

// Get root element
const rootElement = document.getElementById('app');

if (!rootElement) {
    throw new Error('Root element #app not found');
}

// Create React 18 root
const root = ReactDOM.createRoot(rootElement);

// Render application
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);

// Log initialization
console.log('🚀 Documentation Viewer initialized');
