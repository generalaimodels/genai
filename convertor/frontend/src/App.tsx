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

            // Route to docs page for ANY non-empty hash path
            // Show main page only for empty hash or bare "/"
            if (hash && hash !== '/' && hash !== '') {
                setCurrentRoute('docs');
            } else {
                setCurrentRoute('main');
            }
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
