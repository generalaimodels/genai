/**
 * Centralized TypeScript Type Definitions
 * 
 * Single source of truth for all application types.
 * Following best practices for type-first development.
 */

// ============================================
// API Response Types
// ============================================

/** Document heading extracted from markdown */
export interface Heading {
    level: number;
    text: string;
    id: string;
}

/** Document metadata without content */
export interface DocumentMetadata {
    path: string;
    title: string;
    description: string | null;
    modified_at: string;
    size_bytes: number;
    heading_count: number;
}

/** Full document content with parsed HTML */
export interface DocumentContent {
    metadata: DocumentMetadata;
    content_html: string;
    headings: Heading[];
    front_matter: Record<string, unknown>;
}

/** Navigation tree node - folder or document */
export interface NavigationNode {
    name: string;
    path: string | null;
    is_directory: boolean;
    children: NavigationNode[];
}

/** Search result match with context */
export interface SearchMatch {
    text: string;
    heading: string | null;
    line: number;
}

/** Single search result */
export interface SearchResult {
    path: string;
    title: string;
    score: number;
    matches: SearchMatch[];
}

/** API response for document list */
export interface DocumentListResponse {
    documents: DocumentMetadata[];
    total: number;
}

/** API response for search */
export interface SearchResponse {
    results: SearchResult[];
    query: string;
    total: number;
}

// ============================================
// Application State Types
// ============================================

/** Theme variants */
export type Theme = 'light' | 'dark';

/** Application global state */
export interface AppState {
    currentPath: string | null;
    theme: Theme;
    sidebarCollapsed: boolean;
    tocCollapsed: boolean;
    searchOpen: boolean;
}

/** Actions for app state reducer */
export type AppAction =
    | { type: 'SET_PATH'; payload: string | null }
    | { type: 'SET_THEME'; payload: Theme }
    | { type: 'TOGGLE_SIDEBAR' }
    | { type: 'TOGGLE_TOC' }
    | { type: 'SET_SIDEBAR_COLLAPSED'; payload: boolean }
    | { type: 'SET_TOC_COLLAPSED'; payload: boolean }
    | { type: 'OPEN_SEARCH' }
    | { type: 'CLOSE_SEARCH' };

// ============================================
// Component Prop Types
// ============================================

/** Navigation item click handler */
export type NavigateFunction = (path: string) => void;

/** Props for navigation components */
export interface NavigationProps {
    onNavigate: NavigateFunction;
}

/** Props for document viewer */
export interface DocumentViewerProps {
    path: string;
    onHeadingsExtracted?: (headings: Heading[]) => void;
}

/** Props for search modal */
export interface SearchModalProps {
    isOpen: boolean;
    onClose: () => void;
    onNavigate: NavigateFunction;
}

/** Props for table of contents */
export interface TableOfContentsProps {
    headings: Heading[];
}

/** Props for nav item */
export interface NavItemProps {
    node: NavigationNode;
    onNavigate: NavigateFunction;
    level?: number;
}

/** Props for code block */
export interface CodeBlockProps {
    code: string;
    language: string;
    showLineNumbers?: boolean;
}

/** Props for skeleton loader */
export interface SkeletonProps {
    width?: string | number;
    height?: string | number;
    variant?: 'text' | 'rectangular' | 'circular';
    animation?: 'pulse' | 'wave' | 'none';
}

/** Props for modal */
export interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    children: React.ReactNode;
    title?: string;
    size?: 'sm' | 'md' | 'lg' | 'xl';
}

/** Button variants */
export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

/** Props for button */
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: ButtonVariant;
    size?: ButtonSize;
    isLoading?: boolean;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
}

// ============================================
// Utility Types
// ============================================

/** Async operation state */
export interface AsyncState<T> {
    data: T | null;
    loading: boolean;
    error: Error | null;
}

/** Debounced callback signature */
export type DebouncedFunction<T extends (...args: unknown[]) => unknown> = {
    (...args: Parameters<T>): void;
    cancel: () => void;
};
