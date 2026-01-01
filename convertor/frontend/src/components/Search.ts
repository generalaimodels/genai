/**
 * Search Component
 * 
 * Search modal with:
 * - Keyboard shortcut (Cmd/Ctrl + K)
 * - Debounced input
 * - Highlighted results
 * - Keyboard navigation
 */

import { api, SearchResult } from '../api/client';

let selectedIndex = 0;
let results: SearchResult[] = [];
let debounceTimer: number | null = null;

/**
 * Initialize search component
 */
export function initSearch(onNavigate: (path: string) => void): void {
    const modal = document.getElementById('search-modal');
    const trigger = document.getElementById('search-trigger');
    const input = document.getElementById('search-input') as HTMLInputElement;
    const resultsContainer = document.getElementById('search-results');

    if (!modal || !trigger || !input || !resultsContainer) return;

    // Open modal on trigger click
    trigger.addEventListener('click', () => openSearch());

    // Keyboard shortcut: Cmd/Ctrl + K
    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            openSearch();
        }

        // Close on Escape
        if (e.key === 'Escape' && !modal.hidden) {
            closeSearch();
        }
    });

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeSearch();
        }
    });

    // Handle input
    input.addEventListener('input', () => {
        const query = input.value.trim();

        // Clear previous timer
        if (debounceTimer) {
            clearTimeout(debounceTimer);
        }

        if (query.length === 0) {
            renderNoQuery(resultsContainer);
            return;
        }

        // Debounce search
        debounceTimer = window.setTimeout(async () => {
            await performSearch(query, resultsContainer);
        }, 200);
    });

    // Keyboard navigation
    input.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            selectResult(selectedIndex + 1);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            selectResult(selectedIndex - 1);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (results[selectedIndex]) {
                onNavigate(results[selectedIndex].path);
                closeSearch();
            }
        }
    });

    // Click on result
    resultsContainer.addEventListener('click', (e) => {
        const item = (e.target as Element).closest('.search-result-item');
        if (item) {
            const path = item.getAttribute('data-path');
            if (path) {
                onNavigate(path);
                closeSearch();
            }
        }
    });
}

/**
 * Open search modal
 */
function openSearch(): void {
    const modal = document.getElementById('search-modal');
    const input = document.getElementById('search-input') as HTMLInputElement;

    if (!modal || !input) return;

    modal.hidden = false;
    // Small delay for animation
    requestAnimationFrame(() => {
        input.focus();
        input.select();
    });
}

/**
 * Close search modal
 */
function closeSearch(): void {
    const modal = document.getElementById('search-modal');
    const input = document.getElementById('search-input') as HTMLInputElement;
    const resultsContainer = document.getElementById('search-results');

    if (!modal || !input || !resultsContainer) return;

    modal.hidden = true;
    input.value = '';
    results = [];
    selectedIndex = 0;
    renderNoQuery(resultsContainer);
}

/**
 * Perform search
 */
async function performSearch(
    query: string,
    container: Element
): Promise<void> {
    try {
        const response = await api.search(query);
        results = response.results;
        selectedIndex = 0;

        if (results.length === 0) {
            container.innerHTML = `
        <div class="search-no-results">
          No results found for "<strong>${escapeHtml(query)}</strong>"
        </div>
      `;
        } else {
            renderResults(container);
        }
    } catch (error) {
        console.error('Search error:', error);
        container.innerHTML = `
      <div class="search-no-results">
        Search failed. Please try again.
      </div>
    `;
    }
}

/**
 * Render search results
 */
function renderResults(container: Element): void {
    container.innerHTML = results
        .map((result, index) => {
            const excerpt = result.matches[0]?.text || result.title;
            const isSelected = index === selectedIndex;

            return `
        <div class="search-result-item ${isSelected ? 'selected' : ''}" 
             data-path="${escapeHtml(result.path)}"
             data-index="${index}">
          <div class="search-result-title">${escapeHtml(result.title)}</div>
          <div class="search-result-path">${escapeHtml(result.path)}</div>
          <div class="search-result-excerpt">${excerpt}</div>
        </div>
      `;
        })
        .join('');
}

/**
 * Render empty state
 */
function renderNoQuery(container: Element): void {
    container.innerHTML = `
    <div class="search-hint">Type to search...</div>
  `;
}

/**
 * Select result by index
 */
function selectResult(index: number): void {
    const container = document.getElementById('search-results');
    if (!container) return;

    // Clamp index
    if (index < 0) index = results.length - 1;
    if (index >= results.length) index = 0;

    selectedIndex = index;

    // Update UI
    container.querySelectorAll('.search-result-item').forEach((item, i) => {
        item.classList.toggle('selected', i === selectedIndex);
    });

    // Scroll into view
    const selected = container.querySelector('.search-result-item.selected');
    if (selected) {
        selected.scrollIntoView({ block: 'nearest' });
    }
}

/**
 * Escape HTML
 */
function escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
