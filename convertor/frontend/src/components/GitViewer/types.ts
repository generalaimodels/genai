/**
 * Git Viewer TypeScript Interfaces
 * =================================
 * Strict type definitions for Git Pipeline API integration.
 * Enforces type safety across all component boundaries.
 */

/** Repository processing status enumeration */
export type JobStatus =
    | 'pending'
    | 'cloning'
    | 'processing'
    | 'completed'
    | 'failed';

/** API Response wrapper with discriminated union */
export interface ApiResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
}

/** Repository metadata from status endpoint */
export interface RepoMetadata {
    id: string;
    url: string;
    name: string;
    branch: string;
    status: JobStatus;
    total_files: number;
    processed_files: number;
    size_bytes: number;
    last_synced: number;
    created_at: number;
}

/** File tree node (recursive) */
export interface TreeNode {
    name: string;
    path: string;
    type: 'file' | 'directory';
    size?: number;
    children?: TreeNode[];
}

/** Document content from doc endpoint */
export interface Document {
    id: number;
    repo_id: string;
    path: string;
    content: string;
    content_html: string;
    metadata: DocumentMetadata;
    file_type: string;
    size_bytes: number;
    content_hash: string;
    updated_at: number;
}

/** Document metadata (parsed from JSON) */
export interface DocumentMetadata {
    title?: string;
    headings?: string[];
}

/** Process request payload */
export interface ProcessRequest {
    url: string;
    branch?: string;
    depth?: number;
}

/** Cache statistics from /cache/stats */
export interface CacheStats {
    metadata: CacheTier;
    content: CacheTier;
    tree: CacheTier;
}

export interface CacheTier {
    size: number;
    capacity: number;
    hits: number;
    misses: number;
    hit_rate_percent: number;
}

/** Component state for GitViewer */
export interface GitViewerState {
    repoId: string | null;
    status: RepoMetadata | null;
    tree: TreeNode[];
    selectedPath: string | null;
    document: Document | null;
    isLoading: boolean;
    error: string | null;
}

/** Action types for state reducer */
export type GitViewerAction =
    | { type: 'SET_REPO_ID'; payload: string }
    | { type: 'SET_STATUS'; payload: RepoMetadata }
    | { type: 'SET_TREE'; payload: TreeNode[] }
    | { type: 'SELECT_FILE'; payload: string }
    | { type: 'SET_DOCUMENT'; payload: Document }
    | { type: 'SET_LOADING'; payload: boolean }
    | { type: 'SET_ERROR'; payload: string | null }
    | { type: 'RESET' };
