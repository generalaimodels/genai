/**
 * EditorLayout - Main Editor Container with Split-Pane
 * 
 * Architecture:
 * - Split-pane: Monaco Editor (left) + Live Preview (right)
 * - Toolbar: Formatting buttons (bold, italic, headers, etc.)
 * - Status bar: Word count, character count, cursor position
 * - Auto-save: Debounced with visual indicator
 * 
 * State Management:
 * - Local state for editor content
 * - Debounced API calls for live preview (300ms)
 * - Optimistic UI updates for file operations
 * 
 * Performance:
 * - Debounced preview rendering: O(1) cache hits
 * - Atomic file saves: Temp + rename
 * - ETags for conflict detection
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MonacoEditor } from '../MonacoEditor';
import { EditorPreview } from '../EditorPreview';
import './EditorLayout.css';

const API_BASE = 'http://localhost:8000/api/editor';

interface EditorLayoutProps {
    path?: string;
}

interface PreviewData {
    html: string;
    toc: Array<{ text: string; level: number; id: string; line: number }>;
    metadata: {
        word_count: number;
        char_count: number;
        heading_count: number;
        code_block_count: number;
    };
    render_time_ms: number;
    cached: boolean;
}

interface FileMetadata {
    path: string;
    content: string;
    etag: string;
    size_bytes: number;
    modified_at: number;
    author: string | null;
    version: number;
}

export const EditorLayout: React.FC<EditorLayoutProps> = ({ path }) => {

    // Editor state
    const [content, setContent] = useState<string>('');
    const [originalContent, setOriginalContent] = useState<string>('');
    const [etag, setEtag] = useState<string>('');
    const [isDirty, setIsDirty] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [lastSaved, setLastSaved] = useState<Date | null>(null);
    const [filename, setFilename] = useState<string>('New Document'); // Display name
    const [isEditingFilename, setIsEditingFilename] = useState(false); // Editing state

    // Preview state
    const [preview, setPreview] = useState<PreviewData | null>(null);
    const [isPreviewLoading, setIsPreviewLoading] = useState(false);

    // UI state
    const [splitPosition, setSplitPosition] = useState(50); // percentage
    const [showToolbar, setShowToolbar] = useState(true);
    const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });

    // Refs
    const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null);
    const previewDebounceRef = useRef<NodeJS.Timeout | null>(null);
    const editorRef = useRef<any>(null); // Monaco Editor instance

    /**
     * Extract clean filename from path for display
     */
    const extractFilename = (filepath: string | undefined): string => {
        if (!filepath) return 'New Document';
        const segments = filepath.split(/[/\\]/);
        const basename = segments[segments.length - 1];
        return basename.replace(/\.md$/, '');
    };

    /**
     * Load file on mount or path change
     */
    useEffect(() => {
        if (path) {
            setFilename(extractFilename(path));
            loadFile(path);
        } else {
            // New file mode
            setFilename('New Document');
            setContent('# New Document\n\nStart writing...');
            setOriginalContent('');
            setEtag('');
            setIsDirty(false);
        }
    }, [path]);

    /**
     * Load file from API
     */
    const loadFile = async (filePath: string) => {
        try {
            const response = await fetch(`${API_BASE}/files/${encodeURIComponent(filePath)}`);

            if (!response.ok) {
                throw new Error(`Failed to load file: ${response.statusText}`);
            }

            const data: FileMetadata = await response.json();
            setContent(data.content);
            setOriginalContent(data.content);
            setEtag(data.etag);
            setIsDirty(false);

            // Trigger initial preview
            renderPreview(data.content);

        } catch (error) {
            console.error('Error loading file:', error);
            alert(`Failed to load file: ${error}`);
        }
    };

    /**
     * Handle content change from editor
     */
    const handleContentChange = useCallback((newContent: string) => {
        setContent(newContent);
        setIsDirty(newContent !== originalContent);

        // Debounced preview rendering
        if (previewDebounceRef.current) {
            clearTimeout(previewDebounceRef.current);
        }

        previewDebounceRef.current = setTimeout(() => {
            renderPreview(newContent);
        }, 300);

        // Reset auto-save timer
        if (autoSaveTimerRef.current) {
            clearTimeout(autoSaveTimerRef.current);
        }

        // Auto-save after 5 seconds of inactivity
        autoSaveTimerRef.current = setTimeout(() => {
            if (path && newContent !== originalContent) {
                saveFile(false);
            }
        }, 5000);

    }, [originalContent, path]);

    /**
     * Render live preview via API
     */
    const renderPreview = async (markdown: string) => {
        setIsPreviewLoading(true);

        try {
            const response = await fetch(`${API_BASE}/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: markdown })
            });

            if (!response.ok) {
                throw new Error(`Preview failed: ${response.statusText}`);
            }

            const data: PreviewData = await response.json();
            setPreview(data);

        } catch (error) {
            console.error('Error rendering preview:', error);
        } finally {
            setIsPreviewLoading(false);
        }
    };

    /**
     * Save file (manual or auto-save) with Save As support
     */
    const saveFile = async (saveAs: boolean = false) => {
        setIsSaving(true);

        try {
            let targetPath = path;

            // Save As: prompt for new filename
            if (saveAs || !path) {
                const userFilename = prompt(
                    'Enter filename (e.g., notes/my-file or just my-file):',
                    filename !== 'New Document' ? filename : ''
                );

                if (!userFilename) {
                    setIsSaving(false);
                    return; // User canceled
                }

                // Sanitize and ensure .md extension
                let sanitized = userFilename.trim();

                // If no directory specified, put in root data folder
                if (!sanitized.includes('/') && !sanitized.includes('\\')) {
                    sanitized = sanitized;
                }

                // Ensure .md extension
                if (!sanitized.endsWith('.md')) {
                    sanitized = `${sanitized}.md`;
                }

                targetPath = sanitized;
            }

            // API call with ETag for conflict detection
            const response = path && !saveAs
                ? await fetch(`${API_BASE}/files/${encodeURIComponent(path)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content, etag })
                })
                : await fetch(`${API_BASE}/files`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: targetPath, content })
                });

            if (!response.ok) {
                if (response.status === 409) {
                    // Conflict - file modified by another user
                    const shouldOverwrite = confirm(
                        'This file has been modified by another user. Overwrite their changes?'
                    );
                    if (shouldOverwrite) {
                        // Retry without ETag
                        const retryResponse = await fetch(`${API_BASE}/files/${encodeURIComponent(path!)}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ content: content })
                        });
                        if (!retryResponse.ok) {
                            const error = await retryResponse.json();
                            throw new Error(error.detail || `Save failed: ${retryResponse.statusText}`);
                        }
                        const retryData: FileMetadata = await retryResponse.json();
                        setEtag(retryData.etag);
                        setOriginalContent(content);
                        setIsDirty(false);
                        setLastSaved(new Date());
                        return; // Successfully retried and saved
                    } else {
                        // Reload file
                        loadFile(path!);
                        setIsSaving(false);
                        return;
                    }
                } else {
                    const error = await response.json();
                    throw new Error(error.detail || `Save failed: ${response.statusText}`);
                }
            }

            const data: FileMetadata = await response.json();

            // Optimistic UI update
            setEtag(data.etag);
            setOriginalContent(content);
            setIsDirty(false);
            setLastSaved(new Date());

            // Update filename display
            if (targetPath) {
                setFilename(extractFilename(targetPath));
            }

            // Navigate if new file or Save As
            if (!path || saveAs) {
                window.location.hash = `#/editor/${targetPath}`;
            }

        } catch (error) {
            console.error('Save error:', error);
            alert(`Failed to save file: ${error instanceof Error ? error.message : 'Unknown error'}`);
        } finally {
            setIsSaving(false);
        }
    };

    /**
     * Create new file
     */
    const createFile = async (newPath: string, initialContent: string) => {
        try {
            const response = await fetch(`${API_BASE}/files`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    path: newPath,
                    content: initialContent
                })
            });

            if (!response.ok) {
                throw new Error(`Create failed: ${response.statusText}`);
            }

            const data: FileMetadata = await response.json();
            setEtag(data.etag);
            setOriginalContent(initialContent);
            setIsDirty(false);
            setLastSaved(new Date());

            // Navigate to new file
            window.location.hash = `#/editor/${encodeURIComponent(newPath)}`;

        } catch (error) {
            console.error('Error creating file:', error);
            alert(`Failed to create file: ${error}`);
        }
    };

    /**
     * Delete current file
     */
    const deleteFile = async () => {
        if (!path) return;

        const confirmed = confirm(`Are you sure you want to delete "${path}"?`);
        if (!confirmed) return;

        try {
            const response = await fetch(`${API_BASE}/files/${encodeURIComponent(path)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`Delete failed: ${response.statusText}`);
            }

            // Navigate to home or new file
            window.location.hash = '#/editor';

        } catch (error) {
            console.error('Error deleting file:', error);
            alert(`Failed to delete file: ${error}`);
        }
    };

    /**
     * Classic: Get human-readable relative time ("2s ago", "1m ago")
     */
    const getRelativeTime = (date: Date): string => {
        const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);

        if (seconds < 10) return 'just now';
        if (seconds < 60) return `${seconds}s ago`;

        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;

        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;

        return date.toLocaleDateString();
    };

    /**
     * Format text with markdown syntax
     */
    const formatText = (type: string, prefix: string, suffix: string = '') => {
        if (!editorRef.current) return;

        const editor = editorRef.current;
        const selection = editor.getSelection();
        if (!selection) return;

        const selectedText = editor.getModel()?.getValueInRange(selection) || '';
        const wrappedText = `${prefix}${selectedText}${suffix}`;

        editor.executeEdits('', [
            {
                range: selection,
                text: wrappedText,
                forceMoveMarkers: true
            }
        ]);

        // Focus editor after formatting
        editor.focus();
    };

    /**
     * Insert example/demo content to show users how to use a feature
     */
    const insertExample = (type: string) => {
        if (!editorRef.current) return;

        const editor = editorRef.current;
        const position = editor.getPosition();
        if (!position) return;

        const examples: Record<string, string> = {
            'bold': '**bold text**',
            'italic': '*italic text*',
            'code': '`inline code`',
            'link': '[link text](https://example.com)',
            'image': '![image description](https://picsum.photos/800/400)',
            'ul': '- First item\n- Second item\n- Third item',
            'ol': '1. First step\n2. Second step\n3. Third step',
            'quote': '> This is a blockquote\n> It can span multiple lines',
            'h1': '# Main Heading',
            'h2': '## Section Heading',
            'h3': '### Subsection Heading',
            'h4': '#### Minor Heading',
            'h5': '##### Small Heading',
            'h6': '###### Tiny Heading',
            'hr': '\n---\n',
            'code-block': '```python\ndef hello_world():\n    print("Hello, World!")\n```',
            'table': '| Column 1 | Column 2 | Column 3 |\n|----------|----------|----------|\n| Data 1   | Data 2   | Data 3   |\n| Data 4   | Data 5   | Data 6   |',
            'equation-inline': '$E = mc^2$',
            'equation-display': '$$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$'
        };

        const exampleText = examples[type] || '';
        if (!exampleText) return;

        // Insert example at cursor position
        editor.executeEdits('', [{
            range: {
                startLineNumber: position.lineNumber,
                startColumn: position.column,
                endLineNumber: position.lineNumber,
                endColumn: position.column
            },
            text: exampleText,
            forceMoveMarkers: true
        }]);

        // Move cursor to the end of inserted text
        const lines = exampleText.split('\n');
        const lastLine = lines[lines.length - 1];
        editor.setPosition({
            lineNumber: position.lineNumber + lines.length - 1,
            column: position.column + lastLine.length
        });

        editor.focus();
    };
    /**
     * Cleanup timers on unmount
     */
    useEffect(() => {
        return () => {
            if (autoSaveTimerRef.current) {
                clearTimeout(autoSaveTimerRef.current);
            }
            if (previewDebounceRef.current) {
                clearTimeout(previewDebounceRef.current);
            }
        };
    }, []);

    return (
        <div className="editor-layout">
            {/* Header */}
            <div className="editor-header">
                <div className="header-left">
                    <button
                        className="btn-secondary btn-icon"
                        onClick={() => window.location.hash = '/#'}
                        title="Back to documents"
                    >
                        ←
                    </button>
                    {/* Editable filename input */}
                    {isEditingFilename ? (
                        <input
                            type="text"
                            className="filename-input"
                            value={filename}
                            onChange={(e) => setFilename(e.target.value)}
                            onBlur={() => setIsEditingFilename(false)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                    setIsEditingFilename(false);
                                    setIsDirty(true);
                                }
                            }}
                            autoFocus
                            placeholder="Document name"
                        />
                    ) : (
                        <h2
                            className="filename-display"
                            onClick={() => setIsEditingFilename(true)}
                            title="Click to edit filename"
                        >
                            {filename}
                        </h2>
                    )}
                    {isDirty && <span className="dirty-indicator" title="Unsaved changes">●</span>}
                    {/* Premium: Cloud sync status */}
                    {lastSaved && (
                        <span className="sync-status">
                            ✓ Saved {getRelativeTime(lastSaved)}
                        </span>
                    )}
                </div>
                <div className="header-right">
                    <button
                        className="btn-secondary"
                        onClick={() => setShowToolbar(!showToolbar)}
                        style={{
                            background: showToolbar ? 'rgba(212, 175, 55, 0.2)' : 'transparent',
                            color: showToolbar ? '#d4af37' : '#94a3b8'
                        }}
                        title="Toggle toolbar visibility"
                    >
                        VIEW
                    </button>
                    <button
                        className="btn-primary"
                        onClick={() => saveFile(false)}
                        disabled={!isDirty || isSaving}
                        title="Save file (Ctrl+S)"
                    >
                        {isSaving ? 'Saving...' : 'Save'}
                    </button>
                    <button
                        className="btn-secondary"
                        onClick={() => saveFile(true)}
                        title="Save as new file with different name"
                    >
                        Save As
                    </button>
                    {path && (
                        <button className="btn-danger" onClick={deleteFile} title="Delete this file">
                            Delete
                        </button>
                    )}
                </div>
            </div>

            {/* Toolbar */}
            <AnimatePresence>
                {
                    showToolbar && (
                        <motion.div
                            className="editor-toolbar"
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.2 }}
                        >
                            {/* Text Formatting */}
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('bold') : formatText('bold', '**', '**')}
                                title="Bold (Ctrl+B) | Shift+Click for example"
                            >B</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('italic') : formatText('italic', '*', '*')}
                                title="Italic (Ctrl+I) | Shift+Click for example"
                            >I</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('code') : formatText('code', '`', '`')}
                                title="Inline Code (Ctrl+`) | Shift+Click for example"
                            >{'</>'}</button>

                            <div className="toolbar-divider"></div>

                            {/* Links & Media */}
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('link') : formatText('link', '[', '](url)')}
                                title="Insert Link (Ctrl+K) | Shift+Click for example"
                            >🔗</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('image') : formatText('image', '![', '](url)')}
                                title="Insert Image | Shift+Click for example"
                            >🖼️</button>

                            <div className="toolbar-divider"></div>

                            {/* Lists */}
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('ul') : formatText('ul', '- ', '')}
                                title="Bullet List | Shift+Click for example"
                            >≡</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('ol') : formatText('ol', '1. ', '')}
                                title="Numbered List | Shift+Click for example"
                            >⋮</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('quote') : formatText('quote', '> ', '')}
                                title="Blockquote | Shift+Click for example"
                            >"</button>

                            <div className="toolbar-divider"></div>

                            {/* Headers */}
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('h1') : formatText('h1', '# ', '')}
                                title="Heading 1 | Shift+Click for example"
                            >H1</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('h2') : formatText('h2', '## ', '')}
                                title="Heading 2 | Shift+Click for example"
                            >H2</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('h3') : formatText('h3', '### ', '')}
                                title="Heading 3 | Shift+Click for example"
                            >H3</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('h4') : formatText('h4', '#### ', '')}
                                title="Heading 4 | Shift+Click for example"
                            >H4</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('h5') : formatText('h5', '##### ', '')}
                                title="Heading 5 | Shift+Click for example"
                            >H5</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('h6') : formatText('h6', '###### ', '')}
                                title="Heading 6 | Shift+Click for example"
                            >H6</button>

                            <div className="toolbar-divider"></div>

                            {/* Advanced */}
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('hr') : formatText('hr', '\n---\n', '')}
                                title="Horizontal Rule | Shift+Click for example"
                            >─</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('code-block') : formatText('code-block', '```\n', '\n```')}
                                title="Code Block | Shift+Click for example"
                            >{'{ }'}</button>
                            <button
                                onClick={(e) => e.shiftKey ? insertExample('table') : formatText('table', '| Col1 | Col2 |\n|------|------|\n| ', ' |')}
                                title="Table | Shift+Click for example"
                            >⊞</button>

                            <div className="toolbar-divider"></div>

                            {/* Equations */}
                            <button
                                className="equation-btn"
                                onClick={(e) => e.shiftKey ? insertExample('equation-inline') : formatText('equation-inline', '$', '$')}
                                title="Inline Equation (LaTeX) | Shift+Click for example"
                            >Ψ</button>
                            <button
                                className="equation-btn"
                                onClick={(e) => e.shiftKey ? insertExample('equation-display') : formatText('equation-display', '$$\n', '\n$$')}
                                title="Display Equation (LaTeX) | Shift+Click for example"
                            >∑</button>
                        </motion.div>
                    )
                }
            </AnimatePresence >

            {/* Split Pane */}
            < div className="editor-main" >
                <div
                    className="editor-pane"
                    style={{ width: `${splitPosition}%` }}
                >
                    <MonacoEditor
                        value={content}
                        onChange={handleContentChange}
                        onSave={() => saveFile()}
                        onMount={(editor) => { editorRef.current = editor; }}
                        language="markdown"
                        theme="vs-dark"
                    />
                </div>

                {/* Resizable Splitter */}
                <div
                    className="editor-splitter"
                    draggable
                    onDragStart={(e) => {
                        e.dataTransfer.effectAllowed = 'move';
                    }}
                    onDrag={(e) => {
                        if (e.clientX > 0) {
                            const newPosition = (e.clientX / window.innerWidth) * 100;
                            if (newPosition > 20 && newPosition < 80) {
                                setSplitPosition(newPosition);
                            }
                        }
                    }}
                />

                <div
                    className="preview-pane"
                    style={{ width: `${100 - splitPosition}%` }}
                >
                    {preview && (
                        <EditorPreview
                            html={preview.html}
                            toc={preview.toc}
                            isLoading={isPreviewLoading}
                        />
                    )}
                </div>
            </div >

            {/* Premium: Enhanced Status Bar */}
            < div className="editor-status" >
                <div className="status-left">
                    {preview && (
                        <>
                            {/* Classic: Word count & reading time */}
                            <button
                                className="status-metric clickable"
                                onClick={() => {
                                    const line = prompt('Go to line number:');
                                    if (line && editorRef.current) {
                                        const lineNum = parseInt(line);
                                        if (!isNaN(lineNum)) {
                                            editorRef.current.setPosition({ lineNumber: lineNum, column: 1 });
                                            editorRef.current.revealLineInCenter(lineNum);
                                            editorRef.current.focus();
                                        }
                                    }
                                }}
                                title="Click to go to line"
                            >
                                Line {cursorPosition.line}:{cursorPosition.column}
                            </button>
                            <span className="status-separator">•</span>
                            <span className="status-metric">{preview.metadata.word_count} words</span>
                            <span className="status-separator">•</span>
                            <span className="status-metric">{preview.metadata.char_count} characters</span>
                            <span className="status-separator">•</span>
                            <span className="status-metric">
                                ~{Math.ceil(preview.metadata.word_count / 200)} min read
                            </span>
                            <span className="status-separator">•</span>
                            <span className="status-metric">{preview.metadata.heading_count} headings</span>
                            {preview.cached && <span className="cache-indicator">⚡ Cached</span>}
                        </>
                    )}
                </div>
                <div className="status-right">
                    {preview && (
                        <span className="render-time">{preview.render_time_ms.toFixed(1)}ms render</span>
                    )}
                </div>
            </div >
        </div >
    );
};

export default EditorLayout;
