/**
 * DocumentViewer Component
 * 
 * Main document renderer with:
 * - Prism.js syntax highlighting
 * - KaTeX math rendering
 * - Mermaid diagram rendering
 * - Copy buttons for code blocks
 * - Heading anchor links
 * - Loading skeleton
 */

import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-sql';
import 'prismjs/components/prism-css';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-diff';

import { api } from '@/api/client';
import { useTheme } from '@/hooks';
import { DocumentSkeleton } from '@/components/ui';
import type { DocumentContent, Heading } from '@/types';

// Lazy load heavy libraries
let mermaidLoaded = false;
let katexLoaded = false;

// Type declarations for dynamic imports
declare global {
    interface Window {
        katex: {
            renderToString: (latex: string, options: Record<string, unknown>) => string;
        };
        mermaid: {
            initialize: (config: Record<string, unknown>) => void;
            render: (id: string, code: string) => Promise<{ svg: string }>;
        };
    }
}

// ============================================
// Welcome Screen
// ============================================

function WelcomeScreen(): React.ReactElement {
    return (
        <motion.div
            className="welcome"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '50vh',
                textAlign: 'center',
            }}
        >
            <h1 style={{ fontSize: 'var(--text-3xl)', marginBottom: 'var(--space-4)' }}>
                Welcome to Documentation
            </h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-6)' }}>
                Select a document from the sidebar to get started.
            </p>
            <p className="keyboard-hint" style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
                Press <kbd style={{ padding: 'var(--space-1) var(--space-2)', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-primary)' }}>⌘</kbd> + <kbd style={{ padding: 'var(--space-1) var(--space-2)', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-primary)' }}>K</kbd> to search
            </p>
        </motion.div>
    );
}

// ============================================
// Error Screen
// ============================================

interface ErrorScreenProps {
    path: string;
}

function ErrorScreen({ path }: ErrorScreenProps): React.ReactElement {
    return (
        <motion.div
            className="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '50vh',
                textAlign: 'center',
            }}
        >
            <h2 style={{ fontSize: 'var(--text-2xl)', color: 'var(--color-error-500)', marginBottom: 'var(--space-4)' }}>
                Document Not Found
            </h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
                Could not load the requested document.
            </p>
            <code style={{ padding: 'var(--space-2) var(--space-4)', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', color: 'var(--text-muted)' }}>
                {path}
            </code>
        </motion.div>
    );
}

// ============================================
// DocumentViewer Component
// ============================================

interface DocumentViewerProps {
    path: string | null;
    onHeadingsChange?: (headings: Heading[]) => void;
}

export function DocumentViewer({ path, onHeadingsChange }: DocumentViewerProps): React.ReactElement {
    const [docContent, setDocContent] = useState<DocumentContent | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(false);
    const containerRef = useRef<HTMLElement>(null);
    const { isDark } = useTheme();

    // Fetch document when path changes
    // OPTIMIZATION: onHeadingsChange excluded from deps to prevent infinite loop
    // Function ref changes on every parent render, causing re-fetch
    useEffect(() => {
        if (!path) {
            setDocContent(null);
            onHeadingsChange?.([]);
            return;
        }

        async function loadDocumentAsync() {
            setLoading(true);
            setError(false);

            try {
                const doc = await api.getDocument(path!);
                setDocContent(doc);

                // Update page title
                const pageTitle = doc.headings[0]?.text ?? path!.split('/').pop() ?? 'Documentation';
                window.document.title = `${pageTitle} | Docs`;
            } catch (err) {
                console.error('Failed to load document:', err);
                setError(true);
                onHeadingsChange?.([]);
            } finally {
                setLoading(false);
            }
        }

        loadDocumentAsync();
    }, [path]); // FIXED: Removed onHeadingsChange dependency

    // Update headings when content changes - OPTIMIZED
    useEffect(() => {
        if (!docContent) return;

        // Extract headings from content - debounced for performance
        const container = containerRef.current;
        if (!container) return;

        // Use RAF for non-blocking DOM query
        const timeoutId = setTimeout(() => {
            requestAnimationFrame(() => {
                const headingElements = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
                const extractedHeadings: Heading[] = Array.from(headingElements).map((el) => {
                    const level = parseInt(el.tagName[1]);
                    return {
                        id: el.id,
                        text: el.textContent || '',
                        level,
                    };
                });

                onHeadingsChange?.(extractedHeadings);
            });
        }, 100); // Debounce

        return () => clearTimeout(timeoutId);
    }, [docContent]); // FIXED: Removed onHeadingsChange dependency

    // Post-process document content
    useEffect(() => {
        if (!docContent || !containerRef.current) return;

        const container = containerRef.current;

        // Highlight code with Prism
        Prism.highlightAllUnder(container);

        // Add click handlers to heading anchors - OPTIMIZED
        addHeadingAnchors(container);

        // Add copy buttons to code blocks
        addCopyButtons(container);

        // OPTIMIZATION: Lazy load images for performance
        const images = container.querySelectorAll('img[src]');
        if ('IntersectionObserver' in window && images.length > 0) {
            const imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target as HTMLImageElement;
                        if (img.dataset.lazySrc) {
                            img.src = img.dataset.lazySrc;
                            img.removeAttribute('data-lazy-src');
                        }
                        imageObserver.unobserve(img);
                    }
                });
            }, { rootMargin: '50px' });

            images.forEach(img => imageObserver.observe(img));
        }

        // ========================================
        // Math Rendering with Robust Retry Mechanism  
        // ========================================
        // Retry mechanism ensures math renders even on slow KaTeX loading
        let mathAttempts = 0;
        const maxMathAttempts = 8; // Increased for reliability
        let retryTimeout: NodeJS.Timeout | null = null;

        const tryRenderMath = () => {
            console.log(`🔢 Attempting math render (attempt ${mathAttempts + 1}/${maxMathAttempts})...`);
            
            if (typeof (window as any).renderMathOnce === 'function') {
                try {
                    (window as any).renderMathOnce();
                    console.log('✅ Math render triggered successfully');
                } catch (err) {
                    console.error('❌ Math render error:', err);
                }
                mathAttempts++;
                
                // Keep trying a few more times to ensure all equations render
                if (mathAttempts < maxMathAttempts) {
                    const delay = 100 * Math.pow(1.5, mathAttempts - 1); 
                    retryTimeout = setTimeout(tryRenderMath, delay);
                }
            } else if (mathAttempts < maxMathAttempts) {
                console.warn('⚠️ renderMathOnce not available yet, retrying...');
                mathAttempts++;
                const delay = 100 * Math.pow(1.5, mathAttempts - 1);
                retryTimeout = setTimeout(tryRenderMath, delay);
            }
        };

        // Trigger math rendering with multiple strategies
        requestAnimationFrame(() => {
            setTimeout(tryRenderMath, 100); // Initial delay
        });

        const cleanup = () => {
            if (retryTimeout) {
                clearTimeout(retryTimeout);
                retryTimeout = null;
            }
            mathAttempts = maxMathAttempts;
        };

        // Clean up on component unmount
        return cleanup;

        // Render Mermaid diagrams
        renderMermaid(container, isDark);

        // PREMIUM: Delayed scroll to allow transition animation to complete
        // Wait for exit animation (300ms) + partial enter (200ms) = 500ms
        requestAnimationFrame(() => {
            setTimeout(() => {
                window.scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
            }, 500); // Sync with transition duration for premium feel
        });
    }, [docContent]); // FIXED: Removed isDark dependency

    // Show welcome screen if no path
    if (!path) {
        return <WelcomeScreen />;
    }

    // Show loading skeleton
    if (loading) {
        return (
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2 }}
            >
                <DocumentSkeleton />
            </motion.div>
        );
    }

    // Show error screen
    if (error) {
        return <ErrorScreen path={path} />;
    }

    // Show document content  
    if (docContent) {
        return (
            <AnimatePresence mode="wait">
                <motion.article
                    key={path}
                    ref={containerRef}
                    className="document-content"
                    // PREMIUM: Smooth entrance animation with subtle lift
                    initial={{ opacity: 0, y: 20, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    // PREMIUM: Elegant exit animation - fade out with slight drop
                    exit={{ opacity: 0, y: -15, scale: 0.99 }}
                    transition={{
                        // OPTIMIZATION: Extended duration for premium feel (600ms)
                        duration: 0.6,
                        // OPTIMIZATION: Smooth cubic-bezier for natural motion
                        ease: [0.16, 1, 0.3, 1],
                        // GPU acceleration hints
                        opacity: { duration: 0.5 },
                        y: { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
                        scale: { duration: 0.6, ease: [0.16, 1, 0.3, 1] }
                    }}
                    // OPTIMIZATION: Force GPU layer for 60 FPS
                    style={{
                        willChange: 'transform, opacity',
                        backfaceVisibility: 'hidden',
                        perspective: 1000
                    }}
                    dangerouslySetInnerHTML={{ __html: docContent.content_html }}
                />
            </AnimatePresence>
        );
    }

    return <WelcomeScreen />;
}

// ============================================
// Helper Functions
// ============================================

function addHeadingAnchors(container: Element): void {
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');

    headings.forEach((heading) => {
        const id = heading.id;
        if (!id) return;

        // Check if anchor already exists
        if (heading.querySelector('.heading-anchor')) return;

        const anchor = document.createElement('a');
        anchor.className = 'heading-anchor';
        anchor.href = `#${id}`;
        anchor.textContent = '#';
        anchor.setAttribute('aria-hidden', 'true');

        // PREMIUM: Prevent default navigation - use smooth scroll with premium easing
        anchor.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // OPTIMIZATION: Smooth scroll to heading with offset for better visibility
            const headerOffset = 80; // Account for sticky header
            const elementPosition = heading.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });

            // CRITICAL FIX: Preserve full document path when adding heading anchor
            // Before: window.location.hash.split('#')[0] → WRONG (loses file extension)
            // After: window.location.hash → CORRECT (preserves full path)
            // Example: #/data/file.md → #/data/file.md#heading-id (NOT #/data/file#heading-id)
            const fullHash = window.location.hash;
            history.replaceState(null, '', `${fullHash}#${id}`);
        });

        heading.appendChild(anchor);
    });
}

function addCopyButtons(container: Element): void {
    const codeBlocks = container.querySelectorAll('.code-block');

    codeBlocks.forEach((block) => {
        const copyBtn = block.querySelector('.copy-btn');
        if (!copyBtn) return;

        // Remove existing listeners by cloning
        const newBtn = copyBtn.cloneNode(true);
        copyBtn.parentNode?.replaceChild(newBtn, copyBtn);

        newBtn.addEventListener('click', async () => {
            const code = block.querySelector('code');
            if (!code) return;

            try {
                await navigator.clipboard.writeText(code.textContent ?? '');

                // ENHANCED: Visual feedback with color change and animation
                const btnElement = newBtn as HTMLElement;
                btnElement.textContent = '✓ Copied!';
                btnElement.classList.add('copied');
                btnElement.style.transform = 'scale(1.1)';

                // Reset after 2 seconds with smooth transition
                setTimeout(() => {
                    btnElement.textContent = 'Copy';
                    btnElement.classList.remove('copied');
                    btnElement.style.transform = '';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
                // Show error state
                const btnElement = newBtn as HTMLElement;
                btnElement.textContent = '✗ Failed';
                btnElement.style.color = '#ef4444';
                setTimeout(() => {
                    btnElement.textContent = 'Copy';
                    btnElement.style.color = '';
                }, 2000);
            }
        });
    });
}

async function renderMath(container: Element): Promise<void> {
    const mathBlocks = container.querySelectorAll('.math-block, .math-inline');

    if (mathBlocks.length === 0) return;

    // Lazy load KaTeX
    if (!katexLoaded) {
        const katex = await import('katex');
        window.katex = katex.default as unknown as typeof window.katex;
        katexLoaded = true;
    }

    mathBlocks.forEach((el) => {
        const latex = el.getAttribute('data-math') ?? el.textContent ?? '';
        const displayMode = el.classList.contains('math-block');

        try {
            // Clean LaTeX markers
            let cleanLatex = latex
                .replace(/^\$\$/, '')
                .replace(/\$\$$/, '')
                .replace(/^\$/, '')
                .replace(/\$$/, '')
                .trim();

            el.innerHTML = window.katex.renderToString(cleanLatex, {
                displayMode,
                throwOnError: false,
                output: 'html',
            });
        } catch (err) {
            console.error('KaTeX error:', err);
        }
    });
}

async function renderMermaid(container: Element, isDark: boolean): Promise<void> {
    const mermaidBlocks = container.querySelectorAll('.mermaid');

    if (mermaidBlocks.length === 0) return;

    // Lazy load Mermaid
    if (!mermaidLoaded) {
        const mermaid = await import('mermaid');

        mermaid.default.initialize({
            startOnLoad: false,
            theme: isDark ? 'dark' : 'default',
            securityLevel: 'strict',
        });

        window.mermaid = mermaid.default;
        mermaidLoaded = true;
    } else {
        // Re-initialize with current theme
        window.mermaid.initialize({
            startOnLoad: false,
            theme: isDark ? 'dark' : 'default',
            securityLevel: 'strict',
        });
    }

    // Render each diagram
    for (let i = 0; i < mermaidBlocks.length; i++) {
        const block = mermaidBlocks[i];
        if (!block) continue;

        const code = block.textContent ?? '';
        const id = `mermaid-${Date.now()}-${i}`;

        try {
            const { svg } = await window.mermaid.render(id, code);
            block.innerHTML = svg;
        } catch (err) {
            console.error('Mermaid error:', err);
            block.innerHTML = `<pre class="mermaid-error">${code}</pre>`;
        }
    }
}

export default DocumentViewer;
