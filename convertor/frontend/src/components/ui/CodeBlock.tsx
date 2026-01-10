/**
 * CodeBlock Component
 * 
 * VSCode-like code block with premium features.
 * 
 * Features:
 * - Copy to clipboard button
 * - Line highlighting support
 * - Optional line numbers
 * - Language badge
 * - Syntax highlighting (via Prism)
 */

import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';

interface CodeBlockProps {
    code: string;
    language?: string;
    highlightLines?: number[];
    showLineNumbers?: boolean;
    className?: string;
}

export function CodeBlock({
    code,
    language = 'text',
    highlightLines = [],
    showLineNumbers = true,
    className = ''
}: CodeBlockProps): React.ReactElement {
    const [copied, setCopied] = useState(false);

    const copyToClipboard = useCallback(async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    }, [code]);

    const lines = code.split('\n');

    return (
        <div className={`code-block-container ${className}`}>
            {/* Header with language badge and copy button */}
            <div className="code-block-header">
                <span className="language-badge">{language}</span>
                <motion.button
                    className="copy-button"
                    onClick={copyToClipboard}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    transition={{ duration: 0.2 }}
                >
                    {copied ? (
                        <>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <polyline points="20 6 9 17 4 12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                            <span>Copied!</span>
                        </>
                    ) : (
                        <>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" strokeWidth="2" />
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" strokeWidth="2" />
                            </svg>
                            <span>Copy</span>
                        </>
                    )}
                </motion.button>
            </div>

            {/* Code content */}
            <pre className="code-block-pre">
                <code className={`language-${language}`}>
                    {lines.map((line, index) => {
                        const lineNumber = index + 1;
                        const isHighlighted = highlightLines.includes(lineNumber);

                        return (
                            <div
                                key={index}
                                className={`code-line ${isHighlighted ? 'highlighted' : ''}`}
                                data-line={lineNumber}
                            >
                                {showLineNumbers && (
                                    <span className="line-number" aria-hidden="true">
                                        {lineNumber}
                                    </span>
                                )}
                                <span className="line-content">{line || '\n'}</span>
                            </div>
                        );
                    })}
                </code>
            </pre>
        </div>
    );
}

export default CodeBlock;
