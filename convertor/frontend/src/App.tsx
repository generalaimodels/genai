/**
 * App Component
 * 
 * Root application component with hash-based routing
 * Routes:
 * - #/ or empty: MainPage (Neural Frontiers Lab landing)
 * - #/any-file-path: DocumentsPage (shows document content from backend)
 */

import React, { useState, useEffect } from 'react';
import { AppProvider } from '@/context';
import { MainPage } from '@/components/MainPage';
import { DocumentsPage } from '@/components/DocumentsPage';

// ============================================
// App Component - Hash-Based Router
// ============================================

export function App(): React.ReactElement {
    const [currentRoute, setCurrentRoute] = useState<'main' | 'docs'>('main');

    useEffect(() => {
        const handleHashChange = () => {
            const hash = window.location.hash.slice(1); // Remove #

            // CRITICAL FIX: Distinguish between document paths and anchor links
            // Document paths contain '/' (e.g., #/path/to/file.md)
            // Anchor links don't contain '/' (e.g., #heading-id for TOC scrolling)
            const isDocumentRoute = hash.includes('/');

            if (isDocumentRoute) {
                // Hash contains '/' → it's a document path, load it
                setCurrentRoute('docs');
            } else if (!hash || hash === '/') {
                // Empty hash or bare '/' → show main page
                setCurrentRoute('main');
            }
            // else: it's an anchor link (#heading-id), stay on current route
            // This allows TOC clicks to scroll without reloading the document
        };

        // Set initial route
        handleHashChange();

        // Listen for hash changes
        window.addEventListener('hashchange', handleHashChange);
        return () => window.removeEventListener('hashchange', handleHashChange);
    }, []);

    return (
        <AppProvider>
            {currentRoute === 'main' ? <MainPage /> : <DocumentsPage />}
        </AppProvider>
    );
}

export default App;
