/**
 * Theme Toggle Component
 * 
 * Handles dark/light theme switching with:
 * - System preference detection
 * - Persistence in localStorage
 * - Smooth transitions
 */

type Theme = 'light' | 'dark';

/**
 * Initialize theme toggle
 */
export function initThemeToggle(): void {
    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;

    toggle.addEventListener('click', () => {
        const current = getCurrentTheme();
        const next: Theme = current === 'light' ? 'dark' : 'light';
        setTheme(next);
    });

    // Listen for system preference changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', (e) => {
        // Only apply if user hasn't set a preference
        if (!localStorage.getItem('theme')) {
            setTheme(e.matches ? 'dark' : 'light');
        }
    });
}

/**
 * Get current theme
 */
export function getCurrentTheme(): Theme {
    return (document.documentElement.getAttribute('data-theme') as Theme) || 'light';
}

/**
 * Set theme
 */
export function setTheme(theme: Theme): void {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
        metaThemeColor.setAttribute(
            'content',
            theme === 'dark' ? '#0f172a' : '#ffffff'
        );
    }
}

/**
 * Load saved theme or detect system preference
 */
export function loadTheme(): void {
    const saved = localStorage.getItem('theme') as Theme | null;

    if (saved) {
        setTheme(saved);
    } else {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        setTheme(prefersDark ? 'dark' : 'light');
    }
}
