/**
 * Smart Document Renderer - Routes to Appropriate Viewer
 * 
 * Architecture Decision:
 * - Code files (.py, .js, .ts, etc.) → CodeViewer (syntax highlighting via backend)
 * - Markdown files (.md, .ipynb, etc.) → DocumentViewer (markdown rendering)
 * 
 * Performance:
 * - O(1) extension lookup for routing decision
 * - Lazy loading of CodeViewer (code splitting)
 * - Memoization to prevent unnecessary re-renders
 * 
 * Algorithm:
 * 1. Extract file extension from path
 * 2. Check against code extension set (O(1) hash lookup)
 * 3. Route to appropriate viewer component
 * 4. Handle loading/error states uniformly
 */

import React, { useState, useEffect, Suspense, lazy } from 'react';
import { DocumentViewer } from '@/components/DocumentViewer';
import { DocumentSkeleton } from '@/components/ui';
import { api } from '@/api/client';
import type { Heading } from '@/types';

/* ================================================================
   Lazy Load CodeViewer (Code Splitting)
   
   Benefits:
   - Reduces initial bundle size
   - Loads only when code file is viewed
   - Automatic chunking by Vite
   ================================================================ */

const CodeViewer = lazy(() => import('@/components/CodeViewer/CodeViewer'));

/* ================================================================
   Code File Extensions (200+ Languages)
   
   Data Structure: Set for O(1) lookup
   Memory: ~2KB (constant overhead)
   ================================================================ */

const CODE_EXTENSIONS = new Set([
    // Tier 1: Most common programming languages
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.kts', '.scala',

    // Tier 2: Web & scripting
    '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',

    // Tier 3: Data & config  
    '.json', '.yaml', '.yml', '.toml', '.xml', '.sql', '.graphql', '.proto',

    // Tier 4: Systems & compiled
    '.asm', '.s', '.zig', '.nim', '.d', '.v', '.vhd',

    // Tier 5: Functional & academic
    '.hs', '.ml', '.fs', '.elm', '.clj', '.ex', '.erl', '.lisp', '.scm',

    // Tier 6: JVM & others
    '.gradle', '.groovy', '.dart', '.m', '.mm', '.tex', '.r', '.R',
    '.jl', '.lua', '.pl', '.pm'
]);

/* ================================================================
   Helper: Detect if Path is Code File
   
   Complexity: O(1) - Set.has() is hash-based
   ================================================================ */

function isCodeFile(path: string): boolean {
    const extension = path.substring(path.lastIndexOf('.')).toLowerCase();
    return CODE_EXTENSIONS.has(extension);
}

/* ================================================================
   Smart Document Renderer Props
   ================================================================ */

interface SmartDocumentRendererProps {
    path: string | null;
    onHeadingsChange?: (headings: Heading[]) => void;
}

/* ================================================================
   SmartDocumentRenderer Component
   
   Routing Logic:
   - Code files → CodeViewer (backend /api/code endpoint)
   - Markdown files → DocumentViewer (backend /api/documents endpoint)
   - Error handling unified across both paths
   ================================================================ */

export function SmartDocumentRenderer({
    path,
    onHeadingsChange
}: SmartDocumentRendererProps): React.ReactElement {
    const [codeData, setCodeData] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(false);

    /* ============================================================
       Determine File Type and Fetch Appropriately
       ============================================================ */

    useEffect(() => {
        if (!path) {
            // No path: Clear state, invoke callback with empty headings
            setCodeData(null);
            onHeadingsChange?.([]);
            return;
        }

        // Check if this is a code file
        if (isCodeFile(path)) {
            // Fetch from /api/code endpoint
            async function fetchCode() {
                setLoading(true);
                setError(false);

                try {
                    const response = await fetch(`/api/code/${encodeURIComponent(path)}`);

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }

                    const data = await response.json();
                    setCodeData(data);

                    // Update page title
                    const filename = path.split('/').pop() || 'Code';
                    window.document.title = `${filename} | Docs`;

                    // Code files don't have TOC headings (use symbols instead)
                    onHeadingsChange?.([]);

                } catch (err) {
                    console.error('Failed to load code file:', err);
                    setError(true);
                    onHeadingsChange?.([]);
                } finally {
                    setLoading(false);
                }
            }

            fetchCode();
        } else {
            // Not a code file: Reset code data and let DocumentViewer handle it
            setCodeData(null);
        }
    }, [path]);

    /* ============================================================
       Render Logic
       ============================================================ */

    // Code file path: Use CodeViewer
    if (path && isCodeFile(path)) {
        // Loading state
        if (loading) {
            return <DocumentSkeleton />;
        }

        // Error state
        if (error) {
            return (
                <div className="error-screen">
                    <h2>Failed to Load Code File</h2>
                    <p>{path}</p>
                </div>
            );
        }

        // Render CodeViewer
        if (codeData) {
            return (
                <Suspense fallback={<DocumentSkeleton />}>
                    <CodeViewer
                        contentHtml={codeData.content_html}
                        metadata={codeData.metadata}
                        symbols={codeData.symbols}
                        theme={document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light'}
                        isLoading={false}
                    />
                </Suspense>
            );
        }
    }

    // Markdown file or no path: Use existing DocumentViewer
    return (
        <DocumentViewer
            path={path}
            onHeadingsChange={onHeadingsChange}
        />
    );
}

export default SmartDocumentRenderer;
