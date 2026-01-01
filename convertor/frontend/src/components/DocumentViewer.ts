/**
 * Document Viewer Component
 * 
 * Renders parsed markdown with:
 * - Prism.js syntax highlighting
 * - Mermaid diagram rendering
 * - KaTeX math rendering
 * - Copy buttons for code
 * - Heading anchor links
 */

import { api, DocumentContent, Heading } from '../api/client';
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

// Lazy load heavy libraries
let mermaidLoaded = false;
let katexLoaded = false;

/**
 * Load document and render to container
 */
export async function loadDocument(path: string): Promise<Heading[]> {
    const container = document.getElementById('content');
    if (!container) return [];

    // Show loading state
    container.innerHTML = `
    <div class="loading">
      <div class="loading-spinner"></div>
      <p>Loading document...</p>
    </div>
  `;

    try {
        const doc = await api.getDocument(path);
        await renderDocument(container, doc);
        return doc.headings;
    } catch (error) {
        console.error('Failed to load document:', error);
        container.innerHTML = `
      <div class="error">
        <h2>Document Not Found</h2>
        <p>Could not load the requested document.</p>
        <p><code>${path}</code></p>
      </div>
    `;
        return [];
    }
}

/**
 * Render document content
 */
async function renderDocument(
    container: Element,
    doc: DocumentContent
): Promise<void> {
    // Create document wrapper
    container.innerHTML = `
    <div class="document-content">
      ${doc.content_html}
    </div>
  `;

    // Post-process: Add anchor links to headings
    addHeadingAnchors(container);

    // Post-process: Add copy buttons to code blocks
    addCopyButtons(container);

    // Highlight code with Prism
    Prism.highlightAllUnder(container);

    // Render math expressions
    await renderMath(container);

    // Render Mermaid diagrams
    await renderMermaid(container);

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Add anchor links to headings
 */
function addHeadingAnchors(container: Element): void {
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');

    headings.forEach((heading) => {
        const id = heading.id;
        if (!id) return;

        const anchor = document.createElement('a');
        anchor.className = 'heading-anchor';
        anchor.href = `#${id}`;
        anchor.textContent = '#';
        anchor.setAttribute('aria-hidden', 'true');

        heading.appendChild(anchor);
    });
}

/**
 * Add copy buttons to code blocks
 */
function addCopyButtons(container: Element): void {
    const codeBlocks = container.querySelectorAll('.code-block');

    codeBlocks.forEach((block) => {
        const copyBtn = block.querySelector('.copy-btn');
        if (!copyBtn) return;

        copyBtn.addEventListener('click', async () => {
            const code = block.querySelector('code');
            if (!code) return;

            try {
                await navigator.clipboard.writeText(code.textContent || '');
                copyBtn.textContent = 'Copied!';
                copyBtn.classList.add('copied');

                setTimeout(() => {
                    copyBtn.textContent = 'Copy';
                    copyBtn.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
            }
        });
    });
}

/**
 * Render math expressions with KaTeX
 */
async function renderMath(container: Element): Promise<void> {
    const mathBlocks = container.querySelectorAll('.math-block, .math-inline');

    if (mathBlocks.length === 0) return;

    // Lazy load KaTeX
    if (!katexLoaded) {
        const katex = await import('katex');
        window.katex = katex.default;
        katexLoaded = true;
    }

    mathBlocks.forEach((el) => {
        const latex = el.getAttribute('data-math') || el.textContent || '';
        const displayMode = el.classList.contains('math-block');

        try {
            // Remove the $...$ or $$...$$ markers if present
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
            // Keep original content on error
        }
    });
}

/**
 * Render Mermaid diagrams
 */
async function renderMermaid(container: Element): Promise<void> {
    const mermaidBlocks = container.querySelectorAll('.mermaid');

    if (mermaidBlocks.length === 0) return;

    // Lazy load Mermaid
    if (!mermaidLoaded) {
        const mermaid = await import('mermaid');

        // Initialize with theme based on current mode
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        mermaid.default.initialize({
            startOnLoad: false,
            theme: isDark ? 'dark' : 'default',
            securityLevel: 'strict',
        });

        window.mermaid = mermaid.default;
        mermaidLoaded = true;
    }

    // Render each diagram
    for (let i = 0; i < mermaidBlocks.length; i++) {
        const block = mermaidBlocks[i];
        const code = block.textContent || '';
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

/**
 * Re-render Mermaid diagrams (e.g., after theme change)
 */
export async function refreshMermaid(): Promise<void> {
    if (!mermaidLoaded) return;

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    window.mermaid.initialize({
        startOnLoad: false,
        theme: isDark ? 'dark' : 'default',
        securityLevel: 'strict',
    });

    // Re-render is complex; for now, suggest page reload
    // A full implementation would store original code and re-render
}

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
