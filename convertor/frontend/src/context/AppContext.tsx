/**
 * Application Context Provider
 * 
 * Global state management for the documentation viewer.
 * 
 * Features:
 * - Centralized app state
 * - Optimized re-renders with split contexts
 * - Persistent sidebar/TOC states
 * - Navigation management
 */

import React, { createContext, useContext, useReducer, useCallback, useEffect, useRef } from 'react';
import type { AppState, AppAction, NavigationNode, Heading, NavigateFunction } from '@/types';

// ============================================
// State Context
// ============================================

function getInitialState(): AppState {
    // Read initial values from localStorage
    let sidebarCollapsed = false;
    let tocCollapsed = false;

    if (typeof window !== 'undefined') {
        try {
            const storedSidebar = localStorage.getItem('sidebar-collapsed');
            const storedToc = localStorage.getItem('toc-collapsed');
            if (storedSidebar) sidebarCollapsed = JSON.parse(storedSidebar);
            if (storedToc) tocCollapsed = JSON.parse(storedToc);
        } catch {
            // Ignore parsing errors
        }
    }

    return {
        currentPath: null,
        theme: 'light',
        sidebarCollapsed,
        tocCollapsed,
        searchOpen: false,
    };
}

function appReducer(state: AppState, action: AppAction): AppState {
    switch (action.type) {
        case 'SET_PATH':
            return { ...state, currentPath: action.payload };
        case 'SET_THEME':
            return { ...state, theme: action.payload };
        case 'TOGGLE_SIDEBAR':
            return { ...state, sidebarCollapsed: !state.sidebarCollapsed };
        case 'TOGGLE_TOC':
            return { ...state, tocCollapsed: !state.tocCollapsed };
        case 'SET_SIDEBAR_COLLAPSED':
            return { ...state, sidebarCollapsed: action.payload };
        case 'SET_TOC_COLLAPSED':
            return { ...state, tocCollapsed: action.payload };
        case 'OPEN_SEARCH':
            return { ...state, searchOpen: true };
        case 'CLOSE_SEARCH':
            return { ...state, searchOpen: false };
        default:
            return state;
    }
}

// ============================================
// Context Types
// ============================================

interface AppContextValue {
    state: AppState;
    dispatch: React.Dispatch<AppAction>;
    navigation: NavigationNode | null;
    setNavigation: (nav: NavigationNode) => void;
    headings: Heading[];
    setHeadings: (headings: Heading[]) => void;
    navigateTo: NavigateFunction;
}

// ============================================
// Context Creation
// ============================================

const AppContext = createContext<AppContextValue | null>(null);

// ============================================
// Provider Component
// ============================================

interface AppProviderProps {
    children: React.ReactNode;
}

export function AppProvider({ children }: AppProviderProps): React.ReactElement {
    const [state, dispatch] = useReducer(appReducer, undefined, getInitialState);
    const [navigation, setNavigation] = React.useState<NavigationNode | null>(null);
    const [headings, setHeadings] = React.useState<Heading[]>([]);

    // Track if initial mount is complete
    const isInitialized = useRef(false);

    // Sync sidebar state to localStorage (skip initial mount)
    useEffect(() => {
        if (!isInitialized.current) {
            isInitialized.current = true;
            return;
        }
        try {
            localStorage.setItem('sidebar-collapsed', JSON.stringify(state.sidebarCollapsed));
        } catch {
            // Ignore storage errors
        }
    }, [state.sidebarCollapsed]);

    // Sync TOC state to localStorage
    useEffect(() => {
        try {
            localStorage.setItem('toc-collapsed', JSON.stringify(state.tocCollapsed));
        } catch {
            // Ignore storage errors
        }
    }, [state.tocCollapsed]);

    // Navigation handler
    const navigateTo = useCallback<NavigateFunction>((path: string) => {
        // Ensure path starts with '/' for consistency
        const normalizedPath = path.startsWith('/') ? path : `/${path}`;

        // Update URL hash - use '#/' format
        window.location.hash = normalizedPath;

        // Update state
        dispatch({ type: 'SET_PATH', payload: path });

        // Close search if open
        dispatch({ type: 'CLOSE_SEARCH' });
    }, []);

    // Handle hash changes
    useEffect(() => {
        const handleHashChange = () => {
            let hash = window.location.hash.slice(1); // Remove #

            // Remove leading slash if present for consistency
            if (hash.startsWith('/')) {
                hash = hash.substring(1);
            }

            // CRITICAL FIX: Ignore anchor-only hashes (no '/')
            // Only update currentPath for actual document routes
            // Anchor links like #heading-id should not trigger document loading
            const isDocumentPath = hash.includes('/') || hash.includes('.');

            if (isDocumentPath && hash) {
                // It's a document path → update currentPath to load it
                dispatch({ type: 'SET_PATH', payload: hash });
            } else if (!hash) {
                // Empty hash → clear currentPath
                dispatch({ type: 'SET_PATH', payload: null });
            }
            // else: it's an anchor (#heading-id), ignore it, don't change currentPath
        };

        // Set initial path from URL
        handleHashChange();

        window.addEventListener('hashchange', handleHashChange);
        return () => window.removeEventListener('hashchange', handleHashChange);
    }, []);

    // Update body data attribute for TOC collapsed state (CSS matching)
    useEffect(() => {
        document.body.setAttribute('data-toc-collapsed', String(state.tocCollapsed));
    }, [state.tocCollapsed]);

    const value: AppContextValue = {
        state,
        dispatch,
        navigation,
        setNavigation,
        headings,
        setHeadings,
        navigateTo,
    };

    return (
        <AppContext.Provider value={value}>
            {children}
        </AppContext.Provider>
    );
}

// ============================================
// Custom Hook for Context Access
// ============================================

export function useApp(): AppContextValue {
    const context = useContext(AppContext);
    if (!context) {
        throw new Error('useApp must be used within an AppProvider');
    }
    return context;
}

// ============================================
// Selector Hooks for Optimized Re-renders
// ============================================

export function useCurrentPath(): string | null {
    const { state } = useApp();
    return state.currentPath;
}

export function useSidebarState(): [boolean, () => void] {
    const { state, dispatch } = useApp();
    const toggle = useCallback(() => dispatch({ type: 'TOGGLE_SIDEBAR' }), [dispatch]);
    return [state.sidebarCollapsed, toggle];
}

export function useTocState(): [boolean, () => void] {
    const { state, dispatch } = useApp();
    const toggle = useCallback(() => dispatch({ type: 'TOGGLE_TOC' }), [dispatch]);
    return [state.tocCollapsed, toggle];
}

export function useSearchState(): [boolean, () => void, () => void] {
    const { state, dispatch } = useApp();
    const open = useCallback(() => dispatch({ type: 'OPEN_SEARCH' }), [dispatch]);
    const close = useCallback(() => dispatch({ type: 'CLOSE_SEARCH' }), [dispatch]);
    return [state.searchOpen, open, close];
}

export default AppContext;
