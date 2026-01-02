/**
 * TableOfContents Component
 * 
 * Document outline with scroll spy highlighting.
 * Features smooth scrolling and active section tracking.
 */

import React, { useEffect, useCallback, memo } from 'react';
import { motion } from 'framer-motion';
import { useIntersectionObserver } from '@/hooks';
import type { Heading } from '@/types';

// ============================================
// TOC Link Component
// ============================================

interface TocLinkProps {
    heading: Heading;
    isActive: boolean;
    onClick: (id: string) => void;
}

const TocLink = memo(function TocLink({ heading, isActive, onClick }: TocLinkProps): React.ReactElement {
    const handleClick = useCallback(
        (e: React.MouseEvent) => {
            e.preventDefault();
            onClick(heading.id);
        },
        [heading.id, onClick]
    );

    return (
        <motion.a
            className={`toc-link ${isActive ? 'active' : ''}`}
            href={`#${heading.id}`}
            data-id={heading.id}
            data-level={heading.level}
            onClick={handleClick}
            initial={false}
            animate={{
                borderLeftColor: isActive ? 'var(--color-primary-500)' : 'transparent',
                color: isActive ? 'var(--color-primary-600)' : 'var(--text-tertiary)',
            }}
            whileHover={{ color: 'var(--text-primary)' }}
            transition={{ duration: 0.15 }}
            style={{
                display: 'block',
                padding: 'var(--space-1) 0',
                fontSize: heading.level >= 3 ? 'var(--text-xs)' : 'var(--text-sm)',
                borderLeft: '2px solid transparent',
                paddingLeft: heading.level >= 3 ? 'calc(var(--space-4) + var(--space-4))' : 'var(--space-4)',
                textDecoration: 'none',
            }}
        >
            {heading.text}
        </motion.a>
    );
});

// ============================================
// TableOfContents Component
// ============================================

interface TableOfContentsProps {
    headings: Heading[];
}

export function TableOfContents({ headings }: TableOfContentsProps): React.ReactElement | null {
    const { activeId, observe, reset } = useIntersectionObserver({
        rootMargin: '-80px 0px -80% 0px',
        threshold: 0,
    });

    // Set up observations when headings change
    useEffect(() => {
        reset();

        // Small delay to ensure DOM elements exist
        const timer = setTimeout(() => {
            for (const heading of headings) {
                observe(heading.id);
            }
        }, 100);

        return () => {
            clearTimeout(timer);
            reset();
        };
    }, [headings, observe, reset]);

    // Scroll to heading
    const scrollToHeading = useCallback((id: string) => {
        const element = document.getElementById(id);
        if (!element) return;

        // Update URL hash without scrolling
        history.pushState(null, '', `#${id}`);

        // Smooth scroll to element
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, []);

    if (headings.length === 0) {
        return null;
    }

    return (
        <nav id="toc-nav" className="toc-nav">
            {headings.map((heading) => (
                <TocLink
                    key={heading.id}
                    heading={heading}
                    isActive={heading.id === activeId}
                    onClick={scrollToHeading}
                />
            ))}
        </nav>
    );
}

export default TableOfContents;
