/**
 * API Client for Document Server
 * 
 * Typed fetch wrapper with:
 * - Strong TypeScript types
 * - Error handling
 * - Request caching
 */

// API Types
export interface Heading {
    level: number;
    text: string;
    id: string;
}

export interface DocumentMetadata {
    path: string;
    title: string;
    description: string | null;
    modified_at: string;
    size_bytes: number;
    heading_count: number;
}

export interface DocumentContent {
    metadata: DocumentMetadata;
    content_html: string;
    headings: Heading[];
    front_matter: Record<string, unknown>;
}

export interface NavigationNode {
    name: string;
    path: string | null;
    is_directory: boolean;
    children: NavigationNode[];
}

export interface SearchMatch {
    text: string;
    heading: string | null;
    line: number;
}

export interface SearchResult {
    path: string;
    title: string;
    score: number;
    matches: SearchMatch[];
}

export interface DocumentListResponse {
    documents: DocumentMetadata[];
    total: number;
}

export interface SearchResponse {
    results: SearchResult[];
    query: string;
    total: number;
}

// API Error class
export class ApiError extends Error {
    constructor(
        message: string,
        public status: number,
        public detail?: string
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

// SOTA LRU Cache
// - 100 entries max (prevents memory bloat)
// - 60 second TTL with lazy expiration
// - O(1) get/set operations via HashMap + Doubly Linked List
// - Per-document invalidation
import { LRUCache } from '@/utils/LRUCache';
const cache = new LRUCache<unknown>(100, 60);

// Base fetch function
async function apiFetch<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const url = `/api${endpoint}`;

    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    });

    if (!response.ok) {
        let detail: string | undefined;
        try {
            const errorData = await response.json();
            detail = errorData.detail;
        } catch {
            // Ignore JSON parse errors
        }
        throw new ApiError(
            `API request failed: ${response.statusText}`,
            response.status,
            detail
        );
    }

    return response.json() as Promise<T>;
}

// API Methods
export const api = {
    /**
     * List all documents
     */
    async listDocuments(): Promise<DocumentListResponse> {
        const cacheKey = 'documents:list';
        const cached = cache.get(cacheKey) as DocumentListResponse | null;
        if (cached) return cached;

        const data = await apiFetch<DocumentListResponse>('/documents');
        cache.set(cacheKey, data);
        return data;
    },

    /**
     * Get a specific document by path
     */
    async getDocument(path: string, bustCache = false): Promise<DocumentContent> {
        const cacheKey = `documents:${path}`;

        // Skip cache if busting
        if (!bustCache) {
            const cached = cache.get(cacheKey) as DocumentContent | null;
            if (cached) return cached;
        }

        // Add cache-busting timestamp if needed to prevent browser HTTP caching
        const cacheBuster = bustCache ? `?_t=${Date.now()}` : '';
        const data = await apiFetch<DocumentContent>(`/documents/${encodeURIComponent(path)}${cacheBuster}`);
        cache.set(cacheKey, data);
        return data;
    },

    /**
     * Get navigation tree
     */
    async getNavigation(): Promise<NavigationNode> {
        const cacheKey = 'navigation';
        const cached = cache.get(cacheKey) as NavigationNode | null;
        if (cached) return cached;

        const data = await apiFetch<NavigationNode>('/navigation');
        cache.set(cacheKey, data);
        return data;
    },

    /**
     * Search documents
     */
    async search(query: string, limit = 20): Promise<SearchResponse> {
        // Don't cache search results
        return apiFetch<SearchResponse>(
            `/search?q=${encodeURIComponent(query)}&limit=${limit}`
        );
    },

    /**
     * Invalidate specific document from cache
     */
    invalidateDocument(path: string): void {
        cache.delete(`documents:${path}`);
    },

    /**
     * Invalidate all documents matching a pattern
     */
    invalidatePattern(pattern: string | RegExp): number {
        return cache.invalidatePattern(pattern);
    },

    /**
     * Clear entire cache
     */
    clearCache(): void {
        cache.clear();
    },

    /**
     * Get cache statistics
     */
    getCacheStats() {
        return cache.getStats();
    },
};
