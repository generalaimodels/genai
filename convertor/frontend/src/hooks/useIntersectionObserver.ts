/**
 * useIntersectionObserver Hook
 * 
 * Efficient scroll spy implementation using Intersection Observer API.
 * 
 * Features:
 * - Lazy observation setup
 * - Configurable threshold and margins
 * - Automatic cleanup
 * - Performance optimized (no scroll event listeners)
 */

import { useEffect, useRef, useState, useCallback } from 'react';

interface IntersectionObserverOptions {
    root?: Element | null;
    rootMargin?: string;
    threshold?: number | number[];
}

interface UseIntersectionObserverReturn {
    /** Currently visible element IDs */
    visibleIds: Set<string>;
    /** ID of the first visible element (for scroll spy) */
    activeId: string | null;
    /** Function to observe an element by ID */
    observe: (id: string) => void;
    /** Function to unobserve an element by ID */
    unobserve: (id: string) => void;
    /** Function to reset all observations */
    reset: () => void;
}

export function useIntersectionObserver(
    options: IntersectionObserverOptions = {}
): UseIntersectionObserverReturn {
    const {
        root = null,
        rootMargin = '-80px 0px -80% 0px',
        threshold = 0,
    } = options;

    const [visibleIds, setVisibleIds] = useState<Set<string>>(new Set());
    const [activeId, setActiveId] = useState<string | null>(null);
    const observerRef = useRef<IntersectionObserver | null>(null);
    const observedElements = useRef<Map<string, Element>>(new Map());
    const orderedIds = useRef<string[]>([]);

    // Initialize observer
    useEffect(() => {
        observerRef.current = new IntersectionObserver(
            (entries) => {
                setVisibleIds((prev) => {
                    const next = new Set(prev);

                    for (const entry of entries) {
                        const id = entry.target.id;
                        if (entry.isIntersecting) {
                            next.add(id);
                        } else {
                            next.delete(id);
                        }
                    }

                    return next;
                });
            },
            { root, rootMargin, threshold }
        );

        return () => {
            if (observerRef.current) {
                observerRef.current.disconnect();
            }
        };
    }, [root, rootMargin, threshold]);

    // Update active ID when visible IDs change
    useEffect(() => {
        if (visibleIds.size === 0) {
            setActiveId(null);
            return;
        }

        // Find the first visible element based on document order
        for (const id of orderedIds.current) {
            if (visibleIds.has(id)) {
                setActiveId(id);
                return;
            }
        }
    }, [visibleIds]);

    // Observe an element by ID
    const observe = useCallback((id: string) => {
        const element = document.getElementById(id);
        if (!element || !observerRef.current) return;

        observedElements.current.set(id, element);

        // Maintain order based on document position
        if (!orderedIds.current.includes(id)) {
            orderedIds.current.push(id);
            // Sort by document position
            orderedIds.current.sort((a, b) => {
                const elA = document.getElementById(a);
                const elB = document.getElementById(b);
                if (!elA || !elB) return 0;
                return elA.compareDocumentPosition(elB) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
            });
        }

        observerRef.current.observe(element);
    }, []);

    // Unobserve an element
    const unobserve = useCallback((id: string) => {
        const element = observedElements.current.get(id);
        if (!element || !observerRef.current) return;

        observerRef.current.unobserve(element);
        observedElements.current.delete(id);
        orderedIds.current = orderedIds.current.filter((i) => i !== id);
    }, []);

    // Reset all observations
    const reset = useCallback(() => {
        if (observerRef.current) {
            observerRef.current.disconnect();
        }
        observedElements.current.clear();
        orderedIds.current = [];
        setVisibleIds(new Set());
        setActiveId(null);
    }, []);

    return {
        visibleIds,
        activeId,
        observe,
        unobserve,
        reset,
    };
}

export default useIntersectionObserver;
