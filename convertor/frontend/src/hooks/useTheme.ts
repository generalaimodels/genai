/**
 * useTheme Hook
 * 
 * Premium theme management with:
 * - System preference detection
 * - Smooth transitions
 * - LocalStorage persistence
 * - Cross-tab synchronization
 */

import { useCallback, useEffect } from 'react';
import { useLocalStorage } from './useLocalStorage';
import type { Theme } from '@/types';

interface UseThemeReturn {
    theme: Theme;
    setTheme: (theme: Theme) => void;
    toggleTheme: () => void;
    isDark: boolean;
    isLight: boolean;
}

/**
 * Detect system color scheme preference
 */
function getSystemTheme(): Theme {
    if (typeof window === 'undefined') return 'light';
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/**
 * Apply theme to document and update meta theme-color
 */
function applyTheme(theme: Theme): void {
    if (typeof document === 'undefined') return;

    // Apply theme attribute for CSS selectors
    document.documentElement.setAttribute('data-theme', theme);

    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
        metaThemeColor.setAttribute(
            'content',
            theme === 'dark' ? '#0f172a' : '#ffffff'
        );
    }
}

export function useTheme(): UseThemeReturn {
    const [storedTheme, setStoredTheme] = useLocalStorage<Theme | null>('theme', null);

    // Derive actual theme from stored preference or system preference
    const theme: Theme = storedTheme ?? getSystemTheme();

    // Apply theme on mount and when it changes
    useEffect(() => {
        applyTheme(theme);
    }, [theme]);

    // Listen for system preference changes
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

        const handleChange = (event: MediaQueryListEvent) => {
            // Only apply if user hasn't set a manual preference
            if (storedTheme === null) {
                applyTheme(event.matches ? 'dark' : 'light');
            }
        };

        mediaQuery.addEventListener('change', handleChange);
        return () => mediaQuery.removeEventListener('change', handleChange);
    }, [storedTheme]);

    const setTheme = useCallback(
        (newTheme: Theme) => {
            setStoredTheme(newTheme);
        },
        [setStoredTheme]
    );

    const toggleTheme = useCallback(() => {
        const newTheme: Theme = theme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
    }, [theme, setTheme]);

    return {
        theme,
        setTheme,
        toggleTheme,
        isDark: theme === 'dark',
        isLight: theme === 'light',
    };
}

export default useTheme;
