/**
 * Main Application Entry Point
 * 
 * SOTA Markdown Documentation Viewer
 * 
 * Features:
 * - Hash-based routing
 * - Component initialization
 * - Event delegation
 */

import './styles/index.css';
import './styles/components.css';

import { initNavigation, setActiveNavItem } from './components/Navigation';
import { loadDocument } from './components/DocumentViewer';
import { renderToc, clearToc } from './components/TableOfContents';
import { initSearch } from './components/Search';
import { initThemeToggle, loadTheme } from './components/ThemeToggle';

/**
 * Navigate to a document
 */
async function navigateTo(path: string): Promise<void> {
    // Update URL hash
    window.location.hash = path;

    // Update active nav item
    setActiveNavItem(path);

    // Load and render document
    const headings = await loadDocument(path);

    // Update TOC
    renderToc(headings);

    // Update page title
    const title = headings[0]?.text || path.split('/').pop() || 'Documentation';
    document.title = `${title} | Docs`;
}

/**
 * Handle hash changes
 */
function handleHashChange(): void {
    const hash = window.location.hash.slice(1); // Remove #

    if (hash) {
        navigateTo(hash);
    } else {
        // Show welcome screen
        showWelcome();
    }
}

/**
 * Show welcome screen
 */
function showWelcome(): void {
    const content = document.getElementById('content');
    if (content) {
        content.innerHTML = `
      <div class="welcome">
        <h1>Welcome to Documentation</h1>
        <p>Select a document from the sidebar to get started.</p>
        <p class="keyboard-hint">
          Press <kbd>⌘</kbd> + <kbd>K</kbd> to search
        </p>
      </div>
    `;
    }

    clearToc();
    document.title = 'Documentation';
}

/**
 * Initialize sidebar toggle buttons
 */
function initSidebarToggles(): void {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const toc = document.getElementById('toc');
    const tocToggle = document.getElementById('toc-toggle');

    // Restore saved states
    const sidebarCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';
    const tocCollapsed = localStorage.getItem('toc-collapsed') === 'true';

    if (sidebarCollapsed && sidebar) {
        sidebar.classList.add('collapsed');
    }

    if (tocCollapsed && toc) {
        toc.classList.add('collapsed');
        document.body.setAttribute('data-toc-collapsed', 'true');
    }

    // Left sidebar toggle
    sidebarToggle?.addEventListener('click', () => {
        if (sidebar) {
            sidebar.classList.toggle('collapsed');
            const isCollapsed = sidebar.classList.contains('collapsed');
            localStorage.setItem('sidebar-collapsed', String(isCollapsed));
        }
    });

    // Right TOC toggle
    tocToggle?.addEventListener('click', () => {
        if (toc) {
            toc.classList.toggle('collapsed');
            const isCollapsed = toc.classList.contains('collapsed');
            // Add data attribute to body for CSS matching
            document.body.setAttribute('data-toc-collapsed', String(isCollapsed));
            localStorage.setItem('toc-collapsed', String(isCollapsed));
        }
    });
}

/**
 * Initialize application
 */
async function init(): Promise<void> {
    console.log('Initializing Markdown Documentation Viewer...');

    // Load theme
    loadTheme();

    // Initialize theme toggle
    initThemeToggle();

    // Initialize sidebar toggles
    initSidebarToggles();

    // Initialize navigation
    await initNavigation(navigateTo);

    // Initialize search
    initSearch(navigateTo);

    // Handle initial route
    handleHashChange();

    // Listen for hash changes
    window.addEventListener('hashchange', handleHashChange);

    console.log('Application initialized');
}

// Start the application
init().catch((error) => {
    console.error('Failed to initialize application:', error);
});
