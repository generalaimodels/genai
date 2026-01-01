/**
 * Navigation Component
 * 
 * Renders sidebar navigation tree with:
 * - Collapsible folders
 * - Active state indication
 * - Keyboard navigation
 */

import { api, NavigationNode } from '../api/client';

let currentPath: string | null = null;

/**
 * Initialize navigation component
 */
export async function initNavigation(
    onNavigate: (path: string) => void
): Promise<void> {
    const container = document.getElementById('navigation');
    if (!container) return;

    try {
        const navigation = await api.getNavigation();
        renderNavigation(container, navigation.children, onNavigate);
    } catch (error) {
        console.error('Failed to load navigation:', error);
        container.innerHTML = `
      <div class="nav-error">
        Failed to load navigation.
        <button onclick="location.reload()">Retry</button>
      </div>
    `;
    }
}

/**
 * Render navigation tree
 */
function renderNavigation(
    container: Element,
    nodes: NavigationNode[],
    onNavigate: (path: string) => void
): void {
    container.innerHTML = '';

    for (const node of nodes) {
        if (node.is_directory && node.children.length > 0) {
            // Render folder
            const group = document.createElement('div');
            group.className = 'nav-group';

            const title = document.createElement('button');
            title.className = 'nav-group-title';
            title.innerHTML = `
        <svg class="nav-group-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M9 18l6-6-6-6"/>
        </svg>
        ${escapeHtml(node.name)}
      `;
            title.addEventListener('click', () => {
                group.classList.toggle('collapsed');
            });

            const items = document.createElement('div');
            items.className = 'nav-group-items';

            // Recursively render children
            renderNavigation(items, node.children, onNavigate);

            group.appendChild(title);
            group.appendChild(items);
            container.appendChild(group);
        } else if (node.path) {
            // Render document link
            const link = document.createElement('a');
            link.className = 'nav-item';
            link.href = `#${node.path}`;
            link.textContent = node.name;
            link.dataset.path = node.path;

            link.addEventListener('click', (e) => {
                e.preventDefault();
                onNavigate(node.path!);
            });

            container.appendChild(link);
        }
    }
}

/**
 * Update active navigation item
 */
export function setActiveNavItem(path: string): void {
    currentPath = path;

    // Remove active from all items
    document.querySelectorAll('.nav-item.active').forEach((el) => {
        el.classList.remove('active');
    });

    // Add active to matching item
    const activeItem = document.querySelector(`.nav-item[data-path="${path}"]`);
    if (activeItem) {
        activeItem.classList.add('active');

        // Expand parent groups
        let parent = activeItem.parentElement;
        while (parent) {
            if (parent.classList.contains('nav-group')) {
                parent.classList.remove('collapsed');
            }
            parent = parent.parentElement;
        }

        // Scroll into view if needed
        activeItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
