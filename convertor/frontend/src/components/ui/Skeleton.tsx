/**
 * Skeleton Loader Component
 * 
 * Premium skeleton loading placeholders with shimmer animation.
 * Provides visual feedback during async operations.
 */

import React from 'react';
import { motion } from 'framer-motion';
import type { SkeletonProps } from '@/types';

export function Skeleton({
    width = '100%',
    height = '1rem',
    variant = 'text',
    animation = 'wave',
}: SkeletonProps): React.ReactElement {
    const baseStyles: React.CSSProperties = {
        width: typeof width === 'number' ? `${width}px` : width,
        height: typeof height === 'number' ? `${height}px` : height,
        background: 'var(--bg-tertiary)',
        position: 'relative',
        overflow: 'hidden',
    };

    const variantStyles: Record<string, React.CSSProperties> = {
        text: {
            borderRadius: 'var(--radius-sm)',
        },
        rectangular: {
            borderRadius: 'var(--radius-md)',
        },
        circular: {
            borderRadius: '50%',
        },
    };

    if (animation === 'none') {
        return (
            <div
                className="skeleton"
                style={{ ...baseStyles, ...variantStyles[variant] }}
                aria-hidden="true"
            />
        );
    }

    if (animation === 'pulse') {
        return (
            <motion.div
                className="skeleton skeleton-pulse"
                style={{ ...baseStyles, ...variantStyles[variant] }}
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                aria-hidden="true"
            />
        );
    }

    // Wave animation (default)
    return (
        <div
            className="skeleton skeleton-wave"
            style={{ ...baseStyles, ...variantStyles[variant] }}
            aria-hidden="true"
        >
            <motion.div
                style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent)',
                }}
                animate={{ x: ['-100%', '100%'] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
            />
        </div>
    );
}

/**
 * Skeleton Text - Multiple lines of skeleton text
 */
export function SkeletonText({
    lines = 3,
    spacing = 8,
}: {
    lines?: number;
    spacing?: number;
}): React.ReactElement {
    return (
        <div className="skeleton-text" style={{ display: 'flex', flexDirection: 'column', gap: `${spacing}px` }}>
            {Array.from({ length: lines }, (_, i) => (
                <Skeleton
                    key={i}
                    width={i === lines - 1 ? '70%' : '100%'}
                    height="1rem"
                />
            ))}
        </div>
    );
}

/**
 * Document Skeleton - Full document loading placeholder
 */
export function DocumentSkeleton(): React.ReactElement {
    return (
        <div className="document-skeleton" style={{ padding: 'var(--space-6)' }}>
            {/* Title */}
            <Skeleton width="60%" height="2.5rem" variant="rectangular" />

            {/* Subtitle */}
            <div style={{ marginTop: 'var(--space-4)' }}>
                <Skeleton width="40%" height="1.25rem" />
            </div>

            {/* Paragraph 1 */}
            <div style={{ marginTop: 'var(--space-8)' }}>
                <SkeletonText lines={4} />
            </div>

            {/* Heading */}
            <div style={{ marginTop: 'var(--space-8)' }}>
                <Skeleton width="45%" height="1.75rem" />
            </div>

            {/* Paragraph 2 */}
            <div style={{ marginTop: 'var(--space-4)' }}>
                <SkeletonText lines={3} />
            </div>

            {/* Code block */}
            <div style={{ marginTop: 'var(--space-6)' }}>
                <Skeleton width="100%" height="120px" variant="rectangular" />
            </div>

            {/* Paragraph 3 */}
            <div style={{ marginTop: 'var(--space-6)' }}>
                <SkeletonText lines={2} />
            </div>
        </div>
    );
}

export default Skeleton;
