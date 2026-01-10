/**
 * Monaco Editor - Premium CodeEditor Component
 * 
 * Features:
 * - VSCode-quality editing experience
 * - Custom dark theme with gold accents
 * - Markdown-specific IntelliSense
 * - Minimap, line numbers, word wrap
 * - Keyboard shortcuts (Ctrl+S, Ctrl+B, etc.)
 * - Debounced onChange (300ms) for performance
 * 
 * Performance:
 * - Lazy loading: Monaco loads on first mount
 * - Virtual scrolling: Built-in for large files
 * - Syntax highlighting: Worker-based parsing
 */

import React, { useRef, useCallback, useEffect } from 'react';
import Editor, { Monaco, OnMount } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';

interface MonacoEditorProps {
    value: string;
    onChange?: (value: string) => void;
    onSave?: (value: string) => void;
    onMount?: (editor: editor.IStandaloneCodeEditor) => void;
    language?: string;
    theme?: 'vs-dark' | 'vs-light';
    readOnly?: boolean;
    height?: string;
    debounceMs?: number;
}

export const MonacoEditor: React.FC<MonacoEditorProps> = ({
    value,
    onChange,
    onSave,
    onMount,
    language = 'markdown',
    theme = 'vs-dark',
    readOnly = false,
    height = '100%',
    debounceMs = 300
}) => {
    const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
    const monacoRef = useRef<Monaco | null>(null);
    const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

    /**
     * Handle editor mount - Configure custom theme and keybindings
     */
    const handleEditorDidMount: OnMount = useCallback((editor, monaco) => {
        editorRef.current = editor;
        monacoRef.current = monaco;

        // Define custom theme: VSCode Dark+ with gold accents
        monaco.editor.defineTheme('custom-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [
                // Markdown-specific styling
                { token: 'string.link.markdown', foreground: 'd4af37', fontStyle: 'underline' },
                { token: 'keyword.md', foreground: 'd4af37', fontStyle: 'bold' },
                { token: 'emphasis', foreground: 'e8e8e8', fontStyle: 'italic' },
                { token: 'strong', foreground: 'ffffff', fontStyle: 'bold' },
                { token: 'comment', foreground: '6a9955' },
                { token: 'variable', foreground: '9cdcfe' },
            ],
            colors: {
                'editor.background': '#1e1e1e',
                'editor.foreground': '#cccccc',
                'editor.lineHighlightBackground': '#2a2a2a',
                'editor.selectionBackground': '#3a3d41',
                'editorCursor.foreground': '#d4af37',
                'editorLineNumber.foreground': '#858585',
                'editorLineNumber.activeForeground': '#d4af37', // Classic: Gold for active line
                'editor.inactiveSelectionBackground': '#3a3d41',
                'editorWhitespace.foreground': '#404040',
                'editorIndentGuide.background': '#404040',
                'editorIndentGuide.activeBackground': '#707070',
                'scrollbar.shadow': '#000000',
                'scrollbarSlider.background': '#79797966',
                'scrollbarSlider.hoverBackground': '#646464b3',
                'scrollbarSlider.activeBackground': '#bfbfbf66',
            }
        });

        // Apply custom theme
        monaco.editor.setTheme('custom-dark');

        // Configure editor options
        editor.updateOptions({
            fontSize: 15,
            fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace",
            fontLigatures: true,
            lineHeight: 24, // Classic: 1.6 line-height for better readability
            letterSpacing: 0.5,
            minimap: {
                enabled: true,
                maxColumn: 80,
                renderCharacters: true,
                showSlider: 'mouseover'
            },
            scrollbar: {
                verticalScrollbarSize: 10,
                horizontalScrollbarSize: 10,
                useShadows: true,
                verticalHasArrows: false,
                horizontalHasArrows: false
            },
            lineNumbers: 'on',
            renderLineHighlight: 'all',
            cursorBlinking: 'smooth',
            cursorSmoothCaretAnimation: 'on',
            smoothScrolling: true,
            wordWrap: 'on',
            wrappingIndent: 'indent',
            automaticLayout: true,
            formatOnPaste: true,
            formatOnType: true,
            tabSize: 2,
            insertSpaces: true,
            quickSuggestions: {
                other: true,
                comments: false,
                strings: false
            },
            suggestOnTriggerCharacters: true,
            acceptSuggestionOnEnter: 'on',
            snippetSuggestions: 'top'
        });

        // Register Ctrl+S keybinding for save
        // Register save shortcut
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
            if (onSave && editorRef.current) {
                onSave(editorRef.current.getValue());
            }
        });

        // Expose editor instance to parent via onMount callback
        if (onMount) {
            onMount(editor);
        }

        // Register markdown formatting shortcuts
        registerMarkdownKeybindings(editor, monaco);

        // Focus editor
        editor.focus();
    }, [onSave, onMount]);

    /**
     * Register markdown formatting keybindings
     */
    const registerMarkdownKeybindings = (editor: editor.IStandaloneCodeEditor, monaco: Monaco) => {
        // Ctrl+B: Bold
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyB, () => {
            wrapSelection(editor, '**', '**');
        });

        // Ctrl+I: Italic
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyI, () => {
            wrapSelection(editor, '*', '*');
        });

        // Ctrl+`: Inline code
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Backquote, () => {
            wrapSelection(editor, '`', '`');
        });

        // Ctrl+K: Insert link
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyM, () => {
            wrapSelection(editor, '$', '$');
        });

        // Ctrl+1-6: Headers
        for (let i = 1; i <= 6; i++) {
            editor.addCommand(monaco.KeyMod.CtrlCmd | (monaco.KeyCode.Digit0 + i), () => {
                insertHeader(editor, i);
            });
        }
    };

    /**
     * Wrap selected text with prefix and suffix
     */
    const wrapSelection = (editor: editor.IStandaloneCodeEditor, prefix: string, suffix: string) => {
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
    };

    /**
     * Insert header at current line
     */
    const insertHeader = (editor: editor.IStandaloneCodeEditor, level: number) => {
        const position = editor.getPosition();
        if (!position) return;

        const lineContent = editor.getModel()?.getLineContent(position.lineNumber) || '';
        const prefix = '#'.repeat(level) + ' ';

        // Remove existing header prefix if present
        const cleanedContent = lineContent.replace(/^#{1,6}\s*/, '');
        const newContent = prefix + cleanedContent;

        editor.executeEdits('', [
            {
                range: {
                    startLineNumber: position.lineNumber,
                    startColumn: 1,
                    endLineNumber: position.lineNumber,
                    endColumn: lineContent.length + 1
                },
                text: newContent,
                forceMoveMarkers: true
            }
        ]);
    };

    /**
     * Debounced onChange handler
     */
    const handleChange = useCallback((value: string | undefined) => {
        if (!onChange || value === undefined) return;

        // Clear existing timer
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
        }

        // Set new debounced timer
        debounceTimerRef.current = setTimeout(() => {
            onChange(value);
        }, debounceMs);
    }, [onChange, debounceMs]);

    /**
     * Cleanup on unmount
     */
    useEffect(() => {
        return () => {
            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }
        };
    }, []);

    /**
     * Provide imperative API for parent components
     */
    useEffect(() => {
        // Expose editor instance via ref if needed
        // This allows parent components to call editor methods directly
    }, []);

    return (
        <div style={{ height, width: '100%', position: 'relative' }}>
            <Editor
                height={height}
                language={language}
                theme={theme === 'vs-dark' ? 'custom-dark' : 'vs-light'}
                value={value}
                onChange={handleChange}
                onMount={handleEditorDidMount}
                loading={
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        height: '100%',
                        fontSize: '14px',
                        color: '#888'
                    }}>
                        Loading editor...
                    </div>
                }
                options={{
                    readOnly,
                    domReadOnly: readOnly
                }}
            />
        </div>
    );
};

export default MonacoEditor;
