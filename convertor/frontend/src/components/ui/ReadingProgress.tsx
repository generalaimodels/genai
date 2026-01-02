/**
 * ReadingProgress Component
 * 
 * Premium reading progress slider displayed on the left side of content.
 * Features:
 * - Shows document structure with heading hierarchy
 * - Active section highlighting with scroll spy
 * - Smooth scroll navigation
 * - Glassmorphism design with gradient accents
 * - Premium symbols for visual hierarchy
 */

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Heading } from '@/types';

// ============================================
// Premium Symbols by Heading Level
// ============================================

const HEADING_SYMBOLS: Record<number, string> = {
    1: '●', // Filled circle for H1
    2: '◆', // Diamond for H2
    3: '▸', // Triangle for H3
    4: '▹', // Outline triangle for H4
    5: '•', // Small bullet for H5
    6: '·', // Tiny dot for H6
};

// ============================================
// ReadingProgress Component
// ============================================

interface ReadingProgressProps {
    headings: Heading[];
}

export function ReadingProgress({ headings }: ReadingProgressProps): React.ReactElement | null {
    const [activeId, setActiveId] = useState<string>('');
    const [isVisible, setIsVisible] = useState(true);
    const observerRef = useRef<IntersectionObserver | null>(null);

    // Scroll spy - detect active heading
    useEffect(() => {
        if (headings.length === 0) return;

        // Clean up previous observer
        if (observerRef.current) {
            observerRef.current.disconnect();
        }

        const observer = new IntersectionObserver(
            (entries) => {
                // Find the first visible heading
                const visibleEntry = entries.find(entry => entry.isIntersecting);
                if (visibleEntry) {
                    setActiveId(visibleEntry.target.id);
                }
            },
            {
                rootMargin: '-80px 0px -80% 0px', // Trigger when heading is near top
                threshold: [0, 1],
            }
        );

        // Observe all headings
        headings.forEach(({ id }) => {
            const element = document.getElementById(id);
            if (element) {
                observer.observe(element);
            }
        });

        observerRef.current = observer;

        return () => {
            observer.disconnect();
        };
    }, [headings]);

    // Smooth scroll to heading
    const scrollToHeading = useCallback((id: string) => {
        const element = document.getElementById(id);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            setActiveId(id);
        }
    }, []);

    // Don't render if no headings
    if (headings.length === 0) return null;

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.aside
                    className="reading-progress"
                    initial={{ x: -20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    exit={{ x: -20, opacity: 0 }}
                    transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                >
                    <div className="reading-progress-header">
                        <h3>Contents</h3>
                        <button
                            className="reading-progress-close"
                            onClick={() => setIsVisible(false)}
                            aria-label="Hide reading progress"
                        >
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="18" y1="6" x2="6" y2="18" />
                                <line x1="6" y1="6" x2="18" y2="18" />
                            </svg>
                        </button>
                    </div>

                    <nav className="reading-progress-nav">
                        {headings.map((heading) => {
                            const isActive = activeId === heading.id;
                            const symbol = HEADING_SYMBOLS[heading.level] || '·';

                            return (
                                <motion.button
                                    key={heading.id}
                                    className={`reading-progress-item level-${heading.level} ${isActive ? 'active' : ''}`}
                                    onClick={() => scrollToHeading(heading.id)}
                                    whileHover={{ x: 4 }}
                                    transition={{ duration: 0.2 }}
                                >
                                    <span className="reading-progress-symbol">{symbol}</span>
                                    <span className="reading-progress-text">{heading.text}</span>

                                    {isActive && (
                                        <motion.div
                                            className="reading-progress-indicator"
                                            layoutId="active-indicator"
                                            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                                        />
                                    )}
                                </motion.button>
                            );
                        })}
                    </nav>

                    {/* Gradient accent line */}
                    <div className="reading-progress-accent" />
                </motion.aside>
            )}

            {/* Show button when hidden */}
            {!isVisible && (
                <motion.button
                    className="reading-progress-toggle"
                    onClick={() => setIsVisible(true)}
                    initial={{ x: -20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    whileHover={{ scale: 1.05 }}
                >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="3" y1="12" x2="21" y2="12" />
                        <line x1="3" y1="6" x2="21" y2="6" />
                        <line x1="3" y1="18" x2="21" y2="18" />
                    </svg>
                </motion.button>
            )}
        </AnimatePresence>
    );
}

export default ReadingProgress;
