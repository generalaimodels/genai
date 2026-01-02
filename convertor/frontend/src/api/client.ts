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

// Simple in-memory cache
const cache = new Map<string, { data: unknown; timestamp: number }>();
const CACHE_TTL = 60 * 1000; // 1 minute

function getCached<T>(key: string): T | null {
    const entry = cache.get(key);
    if (entry && Date.now() - entry.timestamp < CACHE_TTL) {
        return entry.data as T;
    }
    return null;
}

function setCache<T>(key: string, data: T): void {
    cache.set(key, { data, timestamp: Date.now() });
}

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
        const cached = getCached<DocumentListResponse>(cacheKey);
        if (cached) return cached;

        const data = await apiFetch<DocumentListResponse>('/documents');
        setCache(cacheKey, data);
        return data;
    },

    /**
     * Get a specific document by path
     */
    async getDocument(path: string): Promise<DocumentContent> {
        const cacheKey = `documents:${path}`;
        const cached = getCached<DocumentContent>(cacheKey);
        if (cached) return cached;

        // Use path directly with proper encoding
        const data = await apiFetch<DocumentContent>(`/documents/${encodeURIComponent(path)}`);
        setCache(cacheKey, data);
        return data;
    },

    /**
     * Get navigation tree
     */
    async getNavigation(): Promise<NavigationNode> {
        const cacheKey = 'navigation';
        const cached = getCached<NavigationNode>(cacheKey);
        if (cached) return cached;

        const data = await apiFetch<NavigationNode>('/navigation');
        setCache(cacheKey, data);
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
     * Clear cache
     */
    clearCache(): void {
        cache.clear();
    },
};
