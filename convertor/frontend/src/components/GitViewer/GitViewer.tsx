/**
 * GitViewer Main Container - 3-Column Layout
 * ============================================
 * Implements recommended page architecture:
 * - Top Bar: Repository URL + Clone & Index
 * - Left Panel: Folder directory (280px)
 * - Center Panel: Readability-optimized content (60-70%)
 * - Right Panel: Contextual metadata (280px)
 */

import { useReducer, useCallback, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type {
    GitViewerState,
    GitViewerAction,
    ApiResponse,
    RepoMetadata,
    TreeNode,
    Document,
    DocumentMetadata
} from './types';
import './GitViewer.css';

const API_BASE = '/api/git';

// ============================================
// STATE MANAGEMENT
// ============================================

const initialState: GitViewerState = {
    repoId: null,
    status: null,
    tree: [],
    selectedPath: null,
    document: null,
    isLoading: false,
    error: null
};

function reducer(state: GitViewerState, action: GitViewerAction): GitViewerState {
    switch (action.type) {
        case 'SET_REPO_ID':
            return { ...state, repoId: action.payload, error: null };
        case 'SET_STATUS':
            return { ...state, status: action.payload };
        case 'SET_TREE':
            return { ...state, tree: action.payload };
        case 'SELECT_FILE':
            return { ...state, selectedPath: action.payload, document: null };
        case 'SET_DOCUMENT':
            return { ...state, document: action.payload };
        case 'SET_LOADING':
            return { ...state, isLoading: action.payload };
        case 'SET_ERROR':
            return { ...state, error: action.payload };
        case 'RESET':
            return initialState;
        default:
            return state;
    }
}

// ============================================
// TREE NODE COMPONENT
// ============================================

interface TreeNodeProps {
    node: TreeNode;
    depth: number;
    selectedPath: string | null;
    onSelect: (path: string) => void;
}

function getFileIcon(name: string): string {
    const ext = name.split('.').pop()?.toLowerCase() || '';
    const icons: Record<string, string> = {
        md: 'M', py: 'Py', ts: 'TS', tsx: 'TS',
        js: 'JS', jsx: 'JS', json: '{ }', yaml: 'Y', yml: 'Y'
    };
    return icons[ext] || '📄';
}

function TreeNodeRow({ node, depth, selectedPath, onSelect }: TreeNodeProps) {
    const [isExpanded, setIsExpanded] = useState(false);
    const isSelected = selectedPath === node.path;
    const isDirectory = node.type === 'directory';
    const hasChildren = isDirectory && node.children && node.children.length > 0;

    const handleClick = () => {
        if (isDirectory) {
            setIsExpanded(prev => !prev);
        } else {
            onSelect(node.path);
        }
    };

    const ext = node.name.split('.').pop()?.toLowerCase() || '';

    return (
        <div className="tree-node" style={{ paddingLeft: depth * 12 }}>
            <div
                className={`tree-node__row ${isSelected ? 'tree-node__row--selected' : ''}`}
                onClick={handleClick}
            >
                <svg
                    className={`tree-node__chevron ${isExpanded ? 'tree-node__chevron--expanded' : ''} ${!hasChildren ? 'tree-node__chevron--hidden' : ''}`}
                    viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                >
                    <polyline points="9 18 15 12 9 6" />
                </svg>

                {isDirectory ? (
                    <svg className="tree-node__icon tree-node__icon--folder" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M10 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-8l-2-2z" />
                    </svg>
                ) : (
                    <span className={`tree-node__icon tree-node__icon--${ext}`}>
                        {getFileIcon(node.name)}
                    </span>
                )}

                <span className="tree-node__name" title={node.path}>{node.name}</span>
            </div>

            <AnimatePresence>
                {hasChildren && isExpanded && (
                    <motion.div
                        className="tree-node__children"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.15 }}
                    >
                        {node.children!.map(child => (
                            <TreeNodeRow
                                key={child.path}
                                node={child}
                                depth={depth + 1}
                                selectedPath={selectedPath}
                                onSelect={onSelect}
                            />
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

// ============================================
// MAIN COMPONENT
// ============================================

import { useState } from 'react';

export function GitViewer() {
    const [state, dispatch] = useReducer(reducer, initialState);
    const [url, setUrl] = useState('');
    const pollIntervalRef = useRef<number | null>(null);

    // API Calls
    const submitRepo = useCallback(async (repoUrl: string) => {
        dispatch({ type: 'SET_LOADING', payload: true });
        dispatch({ type: 'SET_ERROR', payload: null });

        try {
            const response = await fetch(`${API_BASE}/process`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: repoUrl })
            });

            const data: ApiResponse<{ job_id: string }> = await response.json();

            if (data.success && data.data) {
                dispatch({ type: 'SET_REPO_ID', payload: data.data.job_id });
                startPolling(data.data.job_id);
            } else {
                dispatch({ type: 'SET_ERROR', payload: data.error || 'Failed to submit' });
            }
        } catch {
            dispatch({ type: 'SET_ERROR', payload: 'Network error' });
        } finally {
            dispatch({ type: 'SET_LOADING', payload: false });
        }
    }, []);

    const fetchStatus = useCallback(async (repoId: string) => {
        try {
            const response = await fetch(`${API_BASE}/${repoId}/status`);
            const data: ApiResponse<RepoMetadata> = await response.json();

            if (data.success && data.data) {
                dispatch({ type: 'SET_STATUS', payload: data.data });

                if (data.data.status === 'completed') {
                    stopPolling();
                    fetchTree(repoId);
                } else if (data.data.status === 'failed') {
                    stopPolling();
                    dispatch({ type: 'SET_ERROR', payload: 'Processing failed' });
                }
            }
        } catch { }
    }, []);

    const fetchTree = useCallback(async (repoId: string) => {
        try {
            const response = await fetch(`${API_BASE}/${repoId}/tree?depth=4`);
            const data: ApiResponse<TreeNode[]> = await response.json();
            if (data.success && data.data) {
                dispatch({ type: 'SET_TREE', payload: data.data });
            }
        } catch { }
    }, []);

    const fetchDocument = useCallback(async (repoId: string, path: string) => {
        dispatch({ type: 'SELECT_FILE', payload: path });
        dispatch({ type: 'SET_LOADING', payload: true });

        try {
            const response = await fetch(`${API_BASE}/${repoId}/doc?path=${encodeURIComponent(path)}`);
            const data: ApiResponse<Document> = await response.json();
            if (data.success && data.data) {
                dispatch({ type: 'SET_DOCUMENT', payload: data.data });
            }
        } catch { }
        finally {
            dispatch({ type: 'SET_LOADING', payload: false });
        }
    }, []);

    // Polling
    const startPolling = useCallback((repoId: string) => {
        stopPolling();
        fetchStatus(repoId);
        pollIntervalRef.current = window.setInterval(() => fetchStatus(repoId), 2000);
    }, [fetchStatus]);

    const stopPolling = useCallback(() => {
        if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
        }
    }, []);

    useEffect(() => () => stopPolling(), [stopPolling]);

    // Handlers
    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (url) submitRepo(url);
    };

    const handleFileSelect = useCallback((path: string) => {
        if (state.repoId) fetchDocument(state.repoId, path);
    }, [state.repoId, fetchDocument]);

    // Derived state
    const isProcessing = state.status?.status === 'cloning' || state.status?.status === 'processing';
    const showContent = state.status?.status === 'completed';
    const sortedTree = useMemo(() => {
        return [...state.tree].sort((a, b) => {
            if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
            return a.name.localeCompare(b.name);
        });
    }, [state.tree]);

    // TOC from document headings
    const toc = useMemo(() => {
        if (!state.document?.metadata) return [];
        const meta = state.document.metadata as DocumentMetadata;
        return meta.headings || [];
    }, [state.document]);

    const breadcrumbs = state.document?.path.split('/').filter(Boolean) || [];

    // ============================================
    // RENDER
    // ============================================

    return (
        <div className={`git-viewer ${state.isLoading ? 'git-viewer--loading' : ''}`}>

            {/* TOP BAR */}
            <header className="git-topbar">
                <div className="git-topbar__brand">
                    <svg className="git-topbar__logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
                    </svg>
                    <span className="git-topbar__title">Git Docs</span>
                </div>

                <form className="git-topbar__url-form" onSubmit={handleSubmit}>
                    <div className="git-topbar__input-wrapper">
                        <input
                            type="text"
                            className="git-topbar__input"
                            placeholder="https://github.com/owner/repository.git"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            disabled={state.isLoading || isProcessing}
                        />
                        <svg className="git-topbar__input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                        </svg>
                    </div>

                    <button
                        type="submit"
                        className="git-topbar__submit"
                        disabled={!url || state.isLoading || isProcessing}
                    >
                        {(state.isLoading || isProcessing) && <span className="git-spinner" />}
                        Clone & Index
                    </button>
                </form>

                <div className="git-topbar__actions">
                    {state.status && (
                        <span className={`git-topbar__status git-topbar__status--${state.status.status}`}>
                            {state.status.status === 'completed' && `${state.status.total_files.toLocaleString()} files`}
                            {state.status.status === 'processing' && `${state.status.processed_files.toLocaleString()} / ${state.status.total_files || '...'}`}
                            {state.status.status === 'cloning' && 'Cloning...'}
                            {state.status.status === 'pending' && 'Pending'}
                        </span>
                    )}
                </div>
            </header>

            {/* ERROR BAR */}
            {state.error && (
                <div className="git-error">
                    <svg className="git-error__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="15" y1="9" x2="9" y2="15" />
                        <line x1="9" y1="9" x2="15" y2="15" />
                    </svg>
                    {state.error}
                </div>
            )}

            {/* LEFT SIDEBAR - File Tree */}
            <aside className="git-sidebar">
                <div className="git-sidebar__header">
                    <span className="git-sidebar__title">Explorer</span>
                </div>
                <div className="git-sidebar__content">
                    {sortedTree.length === 0 ? (
                        <div style={{ padding: '20px', textAlign: 'center', color: 'var(--agion-color-text-muted)', fontSize: '13px' }}>
                            {showContent ? 'No files found' : 'Enter a repository URL to begin'}
                        </div>
                    ) : (
                        sortedTree.map(node => (
                            <TreeNodeRow
                                key={node.path}
                                node={node}
                                depth={0}
                                selectedPath={state.selectedPath}
                                onSelect={handleFileSelect}
                            />
                        ))
                    )}
                </div>
            </aside>

            {/* CENTER CONTENT */}
            <main className="git-content">
                <div className="git-content__wrapper">
                    {!state.document ? (
                        <div className="git-content__empty">
                            <svg className="git-content__empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                <polyline points="14 2 14 8 20 8" />
                                <line x1="16" y1="13" x2="8" y2="13" />
                                <line x1="16" y1="17" x2="8" y2="17" />
                            </svg>
                            <h2 className="git-content__empty-title">Select a Document</h2>
                            <p className="git-content__empty-subtitle">
                                Choose a file from the explorer to view its contents
                            </p>
                        </div>
                    ) : (
                        <motion.div
                            className="git-content__inner"
                            key={state.document.path}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.2 }}
                        >
                            <nav className="git-content__breadcrumb">
                                {breadcrumbs.map((seg, i) => (
                                    <span key={i}>
                                        {i > 0 && <span className="git-content__breadcrumb-separator">/</span>}
                                        <span className={i === breadcrumbs.length - 1 ? 'git-content__breadcrumb-current' : ''}>
                                            {seg}
                                        </span>
                                    </span>
                                ))}
                            </nav>

                            {state.document.content_html ? (
                                <div
                                    className="git-content__markdown"
                                    dangerouslySetInnerHTML={{ __html: state.document.content_html }}
                                />
                            ) : (
                                <pre className="git-content__markdown">
                                    <code>{state.document.content}</code>
                                </pre>
                            )}
                        </motion.div>
                    )}
                </div>
            </main>

            {/* RIGHT DETAILS */}
            <aside className="git-details">
                <div className="git-details__header">
                    <span className="git-details__title">Details</span>
                </div>
                <div className="git-details__content">
                    {state.document ? (
                        <>
                            <div className="git-meta-card">
                                <div className="git-meta-card__label">File</div>
                                <div className="git-meta-card__value git-meta-card__value--gold">
                                    {state.document.path.split('/').pop()}
                                </div>
                            </div>
                            <div className="git-meta-card">
                                <div className="git-meta-card__label">Type</div>
                                <div className="git-meta-card__value">{state.document.file_type.toUpperCase()}</div>
                            </div>
                            <div className="git-meta-card">
                                <div className="git-meta-card__label">Size</div>
                                <div className="git-meta-card__value">
                                    {(state.document.size_bytes / 1024).toFixed(1)} KB
                                </div>
                            </div>

                            {toc.length > 0 && (
                                <div className="git-toc">
                                    <div className="git-toc__title">On This Page</div>
                                    <ul className="git-toc__list">
                                        {toc.slice(0, 10).map((heading, i) => {
                                            const headingId = heading
                                                .toLowerCase()
                                                .replace(/[^\w\s-]/g, '')
                                                .replace(/\s+/g, '-');
                                            return (
                                                <li
                                                    key={i}
                                                    className="git-toc__item"
                                                    onClick={() => {
                                                        const el = document.getElementById(headingId) ||
                                                            document.querySelector(`h1, h2, h3, h4, h5, h6`)?.closest(`[id*="${headingId.slice(0, 10)}"]`) ||
                                                            Array.from(document.querySelectorAll('.git-content__markdown h1, .git-content__markdown h2, .git-content__markdown h3'))
                                                                .find(h => h.textContent?.toLowerCase().includes(heading.toLowerCase().slice(0, 15)));
                                                        if (el) {
                                                            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                                        }
                                                    }}
                                                >
                                                    {heading}
                                                </li>
                                            );
                                        })}
                                    </ul>
                                </div>
                            )}
                        </>
                    ) : (
                        <div style={{ color: 'var(--agion-color-text-muted)', fontSize: '13px', textAlign: 'center', padding: '20px' }}>
                            Select a file to view details
                        </div>
                    )}
                </div>
            </aside>
        </div>
    );
}

export default GitViewer;
