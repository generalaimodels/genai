/**
 * DocumentsPage Component
 * 
 * Main documentation viewing page with:
 * - Left sidebar: File navigation tree
 * - Center: Document content viewer
 * - Right sidebar: Table of contents
 * - Top header: Search and theme toggle
 * 
 * Fully connected to backend API for real documentation browsing.
 */

import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Navigation } from '@/components/Navigation';
import { SmartDocumentRenderer } from '../SmartDocumentRenderer/SmartDocumentRenderer';
import { TableOfContents } from '@/components/TableOfContents';
import { SearchModal } from '@/components/Search';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useApp, useCurrentPath, useSidebarState, useTocState, useSearchState } from '@/context';
import { useWebSocket } from '@/hooks/useWebSocket';
import { api } from '@/api/client';
import type { Heading } from '@/types';

export function DocumentsPage(): React.ReactElement {
    const { navigateTo } = useApp();
    const currentPath = useCurrentPath();
    const [sidebarCollapsed, toggleSidebar] = useSidebarState();
    const [tocCollapsed, toggleToc] = useTocState();
    const [, openSearch] = useSearchState();
    const [headings, setHeadings] = useState<Heading[]>([]);
    const [refreshKey, setRefreshKey] = useState(0);  // Force re-render on file changes
    const [updateNotification, setUpdateNotification] = useState<string | null>(null);

    // WebSocket connection for live updates
    const { isConnected, lastMessage } = useWebSocket();

    // currentPath is now the direct file path from hash (e.g., "graph.md", "folder/file.md")
    const documentPath = currentPath;

    // OPTIMIZATION: Memoize callback to prevent re-render cycles
    // useCallback creates stable function reference across renders
    const handleHeadingsChange = useCallback((newHeadings: Heading[]) => {
        setHeadings(newHeadings);
    }, []); // Empty deps = reference never changes

    // Navigate to document (use path directly)
    const handleNavigate = (path: string) => {
        navigateTo(path);
    };

    // Handle WebSocket file change events
    useEffect(() => {
        if (lastMessage?.type === 'file_changed' && lastMessage.path) {
            const changedPath = lastMessage.path;
            const currentDoc = decodeURIComponent(currentPath);

            console.log('[Auto-Update] File changed:', changedPath, 'Current:', currentDoc);

            // Check if the changed file matches the currently viewed document
            // Handle both exact match and partial match (for nested paths)
            if (currentDoc === changedPath || currentDoc.includes(changedPath) || changedPath.includes(currentDoc)) {
                console.log('[Auto-Update] Current document changed, refreshing...');

                // Clear API cache for this document
                api.clearCache();

                // Show notification based on action
                const actionText = lastMessage.action === 'modified' ? 'updated' : lastMessage.action;
                setUpdateNotification(`Document ${actionText}`);

                // Hide notification after 3 seconds
                setTimeout(() => setUpdateNotification(null), 3000);

                // Handle delete action differently
                if (lastMessage.action === 'deleted') {
                    console.log('[Auto-Update] Document deleted, navigating to home');
                    // Navigate back to home since document no longer exists
                    window.location.hash = '#/';
                } else {
                    // Force re-render of SmartDocumentRenderer by changing key
                    setRefreshKey(Date.now());
                }
            } else {
                // File changed but not currently viewing it
                // Still clear cache to ensure fresh data on next view
                api.clearCache();
            }
        }
    }, [lastMessage, currentPath]);

    return (
        <div className="docs-page" data-sidebar-collapsed={sidebarCollapsed} data-toc-collapsed={tocCollapsed}>
            {/* WebSocket Connection Status */}
            {!isConnected && (
                <div className="connection-status offline">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="3" />
                        <circle cx="12" cy="12" r="10" opacity="0.3" />
                    </svg>
                    <span>Reconnecting to live updates...</span>
                </div>
            )}

            {/* Update Notification */}
            <AnimatePresence>
                {updateNotification && (
                    <motion.div
                        className="update-notification"
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.3 }}
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="20 6 9 17 4 12" />
                        </svg>
                        <span>✓ {updateNotification}</span>
                    </motion.div>
                )}
            </AnimatePresence>
            {/* Top Header */}
            <header className="docs-header">
                <div className="docs-header-left">
                    <button
                        className="docs-header-btn sidebar-toggle"
                        onClick={toggleSidebar}
                        aria-label="Toggle sidebar"
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="3" y1="12" x2="21" y2="12" />
                            <line x1="3" y1="6" x2="21" y2="6" />
                            <line x1="3" y1="18" x2="21" y2="18" />
                        </svg>
                    </button>

                    <a href="#/" className="docs-logo">
                        <img
                            src="/src/asserts/logopng.png"
                            alt="AEGIS AI Logo"
                            className="docs-logo-img"
                            style={{ height: '32px', width: 'auto', marginRight: 'var(--space-3)' }}
                        />
                        <span className="docs-logo-text" style={{
                            fontSize: '1rem',
                            fontWeight: 600,
                            background: 'linear-gradient(to right, #C5A572, #D4B88A)', /* Authentic AEGIS Gold */
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            letterSpacing: '-0.01em'
                        }}>
                            Advanced Engineering for General Intelligence System
                        </span>
                    </a>
                </div>

                <div className="docs-header-right">
                    <button
                        className="docs-header-btn search-btn"
                        onClick={openSearch}
                        aria-label="Search"
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8" />
                            <path d="M21 21l-4.35-4.35" />
                        </svg>
                        <span className="search-btn-label">Search</span>
                        <kbd className="search-kbd">⌘K</kbd>
                    </button>
                    <ThemeToggle />
                </div>
            </header>

            {/* Main Layout */}
            <div className="docs-layout">
                {/* Left Sidebar - Navigation */}
                <AnimatePresence initial={false}>
                    {!sidebarCollapsed && (
                        <motion.aside
                            className="docs-sidebar docs-sidebar-left"
                            initial={{ x: -280, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            exit={{ x: -280, opacity: 0 }}
                            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                        >
                            <div className="docs-sidebar-header">
                                <h2>Navigation</h2>
                                <button
                                    className="sidebar-header-btn"
                                    onClick={toggleSidebar}
                                    aria-label="Minimize navigation"
                                >
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <polyline points="15 18 9 12 15 6" />
                                    </svg>
                                </button>
                            </div>
                            <div className="docs-sidebar-content">
                                <Navigation />
                            </div>
                        </motion.aside>
                    )}
                </AnimatePresence>



                {/* Main Content Area */}
                <main className="docs-main">
                    <div className="docs-content">
                        <SmartDocumentRenderer
                            key={refreshKey}  // Force re-render on file changes
                            path={documentPath}
                            onHeadingsChange={handleHeadingsChange}
                        />
                    </div>
                </main>

                {/* Right Sidebar - Table of Contents */}
                <AnimatePresence initial={false}>
                    {!tocCollapsed && headings.length > 0 && (
                        <motion.aside
                            className="docs-sidebar docs-sidebar-right"
                            initial={{ x: 280, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            exit={{ x: 280, opacity: 0 }}
                            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                        >
                            <div className="docs-sidebar-header">
                                <h2>On This Page</h2>
                                <button
                                    className="toc-close-btn"
                                    onClick={toggleToc}
                                    aria-label="Close table of contents"
                                >
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <line x1="18" y1="6" x2="6" y2="18" />
                                        <line x1="6" y1="6" x2="18" y2="18" />
                                    </svg>
                                </button>
                            </div>
                            <div className="docs-sidebar-content">
                                <TableOfContents headings={headings} />
                            </div>
                        </motion.aside>
                    )}
                </AnimatePresence>

                {/* TOC Toggle Button (when collapsed) */}
                {tocCollapsed && headings.length > 0 && (
                    <motion.button
                        className="toc-toggle-btn"
                        onClick={toggleToc}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        aria-label="Show table of contents"
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="3" y1="12" x2="21" y2="12" />
                            <line x1="3" y1="6" x2="21" y2="6" />
                            <line x1="3" y1="18" x2="21" y2="18" />
                        </svg>
                    </motion.button>
                )}
            </div>

            {/* Search Modal */}
            <SearchModal onNavigate={handleNavigate} />
        </div>
    );
}

export default DocumentsPage;
