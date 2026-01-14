/**
 * DocPreview Component
 * =====================
 * Document content viewer with markdown rendering.
 * Shows breadcrumb navigation and formatted content.
 */

import { memo } from 'react';
import { motion } from 'framer-motion';
import type { Document } from './types';

interface DocPreviewProps {
    document: Document | null;
    isLoading: boolean;
}

/** Parse path into breadcrumb segments */
function parseBreadcrumb(path: string): string[] {
    return path.split('/').filter(Boolean);
}

export const DocPreview = memo(function DocPreview({
    document,
    isLoading
}: DocPreviewProps) {

    if (isLoading) {
        return (
            <motion.div
                className="git-card doc-preview"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
            >
                <div className="git-card__header">
                    <svg className="git-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                    </svg>
                    <h2 className="git-card__title">Document</h2>
                </div>
                <div className="git-card__content doc-preview__content">
                    <div className="doc-preview__empty">
                        <div className="processing-status__spinner" />
                        <p className="doc-preview__empty-subtitle">Loading document...</p>
                    </div>
                </div>
            </motion.div>
        );
    }

    if (!document) {
        return (
            <motion.div
                className="git-card doc-preview"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.2 }}
            >
                <div className="git-card__header">
                    <svg className="git-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                    </svg>
                    <h2 className="git-card__title">Document</h2>
                </div>
                <div className="git-card__content doc-preview__content">
                    <div className="doc-preview__empty">
                        <svg className="doc-preview__empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                            <line x1="16" y1="13" x2="8" y2="13" />
                            <line x1="16" y1="17" x2="8" y2="17" />
                            <polyline points="10 9 9 9 8 9" />
                        </svg>
                        <h3 className="doc-preview__empty-title">No Document Selected</h3>
                        <p className="doc-preview__empty-subtitle">
                            Select a file from the tree to view its contents
                        </p>
                    </div>
                </div>
            </motion.div>
        );
    }

    const breadcrumbs = parseBreadcrumb(document.path);
    const title = document.metadata?.title || breadcrumbs[breadcrumbs.length - 1] || 'Document';

    return (
        <motion.div
            className="git-card doc-preview"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            key={document.path}
        >
            <div className="git-card__header">
                <svg className="git-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                </svg>
                <h2 className="git-card__title">{title}</h2>
            </div>

            <div className="git-card__content doc-preview__content">
                {/* Breadcrumb */}
                <nav className="doc-preview__breadcrumb">
                    {breadcrumbs.map((segment, index) => (
                        <span key={index}>
                            {index > 0 && <span className="doc-preview__breadcrumb-separator">/</span>}
                            <span className={index === breadcrumbs.length - 1 ? 'doc-preview__breadcrumb-current' : ''}>
                                {segment}
                            </span>
                        </span>
                    ))}
                </nav>

                {/* Content */}
                {document.content_html ? (
                    <div
                        className="doc-preview__markdown"
                        dangerouslySetInnerHTML={{ __html: document.content_html }}
                    />
                ) : (
                    <pre className="doc-preview__markdown">
                        <code>{document.content}</code>
                    </pre>
                )}
            </div>
        </motion.div>
    );
});
