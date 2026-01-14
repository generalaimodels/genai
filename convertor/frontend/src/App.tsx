/**
 * App Component
 * 
 * Root application component with hash-based routing
 * Routes:
 * - #/ or empty: MainPage (Neural Frontiers Lab landing)
 * - #/editor or #/editor/path: EditorLayout (markdown editor)
 * - #/git: GitViewer (GitHub repository documentation viewer)
 * - #/any-file-path: DocumentsPage (shows document content from backend)
 */

import React, { useState, useEffect } from 'react';
import { AppProvider } from '@/context';
import { MainPage } from '@/components/MainPage';
import { DocumentsPage } from '@/components/DocumentsPage';
import { EditorLayout } from '@/components/EditorLayout';
import { GitViewer } from '@/components/GitViewer';

// ============================================
// App Component - Hash-Based Router
// ============================================

type RouteType = 'main' | 'docs' | 'editor' | 'git';

export function App(): React.ReactElement {
    const [currentRoute, setCurrentRoute] = useState<RouteType>('main');
    const [editorPath, setEditorPath] = useState<string | undefined>(undefined);

    useEffect(() => {
        const handleHashChange = () => {
            const hash = window.location.hash.slice(1); // Remove #

            // Check if it's the editor route
            if (hash.startsWith('/editor')) {
                setCurrentRoute('editor');
                // Extract file path if present (e.g., #/editor/notes/file.md)
                const pathMatch = hash.match(/^\/editor\/?(.*)$/);
                setEditorPath(pathMatch?.[1] || undefined);
            }
            // Check if it's the git viewer route
            else if (hash === '/git' || hash.startsWith('/git')) {
                setCurrentRoute('git');
            }
            // CRITICAL FIX: Distinguish between document paths and anchor links
            // Document paths contain '/' (e.g., #/path/to/file.md)
            // Anchor links don't contain '/' (e.g., #heading-id for TOC scrolling)
            else {
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
            }
        };

        // Set initial route
        handleHashChange();

        // Listen for hash changes
        window.addEventListener('hashchange', handleHashChange);
        return () => window.removeEventListener('hashchange', handleHashChange);
    }, []);

    // Render appropriate component based on route
    const renderRoute = () => {
        switch (currentRoute) {
            case 'editor':
                return <EditorLayout path={editorPath} />;
            case 'git':
                return <GitViewer />;
            case 'docs':
                return <DocumentsPage />;
            case 'main':
            default:
                return <MainPage />;
        }
    };

    return (
        <AppProvider>
            {renderRoute()}
        </AppProvider>
    );
}

export default App;
