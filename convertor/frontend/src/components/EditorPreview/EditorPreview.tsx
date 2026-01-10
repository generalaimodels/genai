/**
 * EditorPreview - Rock-Solid Live Preview with Error Resilience
 * 
 * Features:
 * - Bulletproof KaTeX/Prism initialization with retries
 * - Error boundaries for graceful degradation
 * - Resize/zoom event handling
 * - Stable rendering regardless of DOM updates
 * 
 * Performance:
 * - Debounced rendering for smooth updates
 * - Efficient re-render prevention
 * - Memory-safe cleanup
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './EditorPreview.css';

interface EditorPreviewProps {
    html: string;
    toc: Array<{ text: string; level: number; id: string; line: number }>;
    isLoading?: boolean;
    onScrollSync?: (scrollPercentage: number) => void;
}

export const EditorPreview: React.FC<EditorPreviewProps> = React.memo((({
    html,
    toc,
    isLoading = false,
    onScrollSync
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [renderKey, setRenderKey] = useState(0);
    const [renderError, setRenderError] = useState<string | null>(null);
    const renderTimerRef = useRef<NodeJS.Timeout | null>(null);
    const initAttemptsRef = useRef<number>(0);

    /**
     * CRITICAL: Robust library initialization with retry logic
     * Ensures KaTeX, Prism, and Mermaid are ready before use
     */
    const ensureLibrariesLoaded = useCallback(async (): Promise<boolean> => {
        const maxAttempts = 10;
        const retryDelay = 100;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const katexReady = typeof window !== 'undefined' && window.katex && typeof window.renderMathInElement === 'function';
            const prismReady = typeof window !== 'undefined' && window.Prism && typeof window.Prism.highlightAll === 'function';

            if (katexReady && prismReady) {
                console.log(`✓ Libraries ready on attempt ${attempt + 1}`);
                return true;
            }

            if (attempt < maxAttempts - 1) {
                await new Promise(resolve => setTimeout(resolve, retryDelay));
            }
        }

        console.warn('⚠️ Libraries not fully loaded after retries');
        return false;
    }, []);

    /**
     * CRITICAL: Stable math rendering with comprehensive error handling
     */
    const renderMathSafely = useCallback((container: HTMLElement) => {
        try {
            // Check for renderMathInElement (auto-render extension)
            if (typeof window.renderMathInElement === 'function') {
                window.renderMathInElement(container, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false },
                        { left: '\\[', right: '\\]', display: true },
                        { left: '\\(', right: '\\)', display: false }
                    ],
                    throwOnError: false,
                    errorColor: '#cc0000',
                    strict: false,
                    trust: false,
                    macros: {
                        "\\RR": "\\mathbb{R}",
                        "\\NN": "\\mathbb{N}",
                        "\\ZZ": "\\mathbb{Z}",
                        "\\QQ": "\\mathbb{Q}"
                    }
                });
                return true;
            }

            // Fallback: Manual rendering with data-math attributes
            if (window.katex) {
                // Inline math
                const inlineMath = container.querySelectorAll('.math-inline[data-math]');
                inlineMath.forEach((element) => {
                    try {
                        const mathText = element.getAttribute('data-math') || '';
                        window.katex.render(mathText, element as HTMLElement, {
                            throwOnError: false,
                            displayMode: false,
                            strict: false,
                            trust: false
                        });
                    } catch (err) {
                        console.error('KaTeX inline error:', err);
                        (element as HTMLElement).textContent = `[Math Error: ${mathText}]`;
                    }
                });

                // Display math
                const displayMath = container.querySelectorAll('.math-block[data-math]');
                displayMath.forEach((element) => {
                    try {
                        const mathText = element.getAttribute('data-math') || '';
                        window.katex.render(mathText, element as HTMLElement, {
                            throwOnError: false,
                            displayMode: true,
                            strict: false,
                            trust: false
                        });
                    } catch (err) {
                        console.error('KaTeX display error:', err);
                        (element as HTMLElement).textContent = `[Math Error: ${mathText}]`;
                    }
                });

                return true;
            }

            return false;
        } catch (err) {
            console.error('Math rendering failed:', err);
            return false;
        }
    }, []);

    /**
     * CRITICAL: Stable syntax highlighting with error handling
     */
    const highlightCodeSafely = useCallback((container: HTMLElement) => {
        try {
            if (window.Prism) {
                // Use highlightAllUnder for scoped highlighting
                if (typeof window.Prism.highlightAllUnder === 'function') {
                    window.Prism.highlightAllUnder(container);
                } else if (typeof window.Prism.highlightAll === 'function') {
                    window.Prism.highlightAll();
                }
                return true;
            }
            return false;
        } catch (err) {
            console.error('Prism highlighting failed:', err);
            return false;
        }
    }, []);

    /**
     * CRITICAL: Stable Mermaid rendering
     */
    const renderMermaidSafely = useCallback(async (container: HTMLElement) => {
        try {
            if (window.mermaid) {
                const mermaidElements = container.querySelectorAll('.mermaid, .language-mermaid');
                if (mermaidElements.length > 0) {
                    await window.mermaid.run({
                        nodes: Array.from(mermaidElements) as HTMLElement[],
                        suppressErrors: true
                    });
                }
                return true;
            }
            return false;
        } catch (err) {
            console.error('Mermaid rendering failed:', err);
            return false;
        }
    }, []);

    /**
     * MAIN RENDERING PIPELINE - Production-grade with error recovery
     */
    const performRendering = useCallback(async () => {
        if (!containerRef.current) return;

        try {
            setRenderError(null);

            // Step 1: Ensure libraries are loaded
            const librariesReady = await ensureLibrariesLoaded();

            if (!librariesReady) {
                setRenderError('Rendering libraries still loading...');
                // Retry after delay
                if (initAttemptsRef.current < 5) {
                    initAttemptsRef.current++;
                    setTimeout(() => performRendering(), 500);
                }
                return;
            }

            initAttemptsRef.current = 0;

            const container = containerRef.current;

            // Step 2: Syntax highlighting (fast, do first)
            highlightCodeSafely(container);

            // Step 3: Math rendering (slower, but critical)
            const mathRendered = renderMathSafely(container);

            // Step 4: Mermaid diagrams (async, slowest)
            await renderMermaidSafely(container);

            if (!mathRendered) {
                console.warn('Math rendering incomplete - KaTeX may not be loaded');
            }

        } catch (err) {
            console.error('Rendering error:', err);
            setRenderError(`Rendering failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
        }
    }, [ensureLibrariesLoaded, renderMathSafely, highlightCodeSafely, renderMermaidSafely]);

    /**
     * Trigger rendering when HTML changes
     */
    useEffect(() => {
        // Clear any pending renders
        if (renderTimerRef.current) {
            clearTimeout(renderTimerRef.current);
        }

        // Debounce rendering to avoid excessive updates
        renderTimerRef.current = setTimeout(() => {
            performRendering();
        }, 100);

        // Force re-render on HTML change
        setRenderKey(prev => prev + 1);

        return () => {
            if (renderTimerRef.current) {
                clearTimeout(renderTimerRef.current);
            }
        };
    }, [html, performRendering]);

    /**
     * CRITICAL: Re-render on window resize/zoom events
     * This fixes the "zoom breaks rendering" issue
     */
    useEffect(() => {
        const handleResize = () => {
            // Re-trigger rendering on resize/zoom
            performRendering();
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [performRendering]);

    /**
     * Handle scroll synchronization
     */
    const handleScroll = useCallback(() => {
        if (!containerRef.current || !onScrollSync) return;

        const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
        const scrollPercentage = scrollTop / (scrollHeight - clientHeight);

        onScrollSync(scrollPercentage);
    }, [onScrollSync]);

    return (
        <div className="editor-preview">
            <div className="preview-header">
                <h3>Live Preview</h3>
                {isLoading && (
                    <motion.div
                        className="preview-loading"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        Rendering...
                    </motion.div>
                )}
                {renderError && (
                    <div className="preview-error">
                        {renderError}
                    </div>
                )}
            </div>

            <div
                ref={containerRef}
                className="preview-content"
                onScroll={handleScroll}
            >
                <AnimatePresence mode="wait">
                    <motion.div
                        key={renderKey}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.15 }}
                        dangerouslySetInnerHTML={{ __html: html }}
                    />
                </AnimatePresence>
            </div>

            {/* Table of Contents Minimap */}
            {toc.length > 0 && (
                <div className="preview-toc-minimap">
                    <h4>Sections</h4>
                    <ul>
                        {toc.map((heading, index) => (
                            <li
                                key={`${heading.id}-${index}`}
                                style={{ paddingLeft: `${(heading.level - 1) * 12}px` }}
                            >
                                <a
                                    href={`#${heading.id}`}
                                    onClick={(e) => {
                                        e.preventDefault();
                                        const target = containerRef.current?.querySelector(`[id="${heading.id}"]`);
                                        target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                    }}
                                >
                                    {heading.text}
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}));

EditorPreview.displayName = 'EditorPreview';

export default EditorPreview;

// Extended Window interface
declare global {
    interface Window {
        Prism?: {
            highlightAll?: () => void;
            highlightAllUnder?: (element: HTMLElement) => void;
        };
        katex?: {
            render: (tex: string, element: HTMLElement, options?: any) => void;
        };
        renderMathInElement?: (element: HTMLElement, options?: any) => void;
        mermaid?: {
            init?: (config?: any, nodes?: any) => void;
            run?: (config?: { nodes?: HTMLElement[]; suppressErrors?: boolean }) => Promise<void>;
        };
    }
}
