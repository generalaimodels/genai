/**
 * Search Modal Component
 * 
 * Premium search experience with:
 * - Keyboard shortcuts (⌘K / Ctrl+K)
 * - Debounced search
 * - Arrow key navigation
 * - Framer Motion animations
 */

import React, { useState, useEffect, useCallback, useRef, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '@/api/client';
import { useSearchState } from '@/context';
import { useDebounce } from '@/hooks';
import { Kbd } from '@/components/ui';
import type { SearchResult, NavigateFunction } from '@/types';

// ============================================
// Search Result Item
// ============================================

interface SearchResultItemProps {
    result: SearchResult;
    isSelected: boolean;
    onClick: () => void;
    onMouseEnter: () => void;
}

const SearchResultItem = memo(function SearchResultItem({
    result,
    isSelected,
    onClick,
    onMouseEnter,
}: SearchResultItemProps): React.ReactElement {
    const excerpt = result.matches[0]?.text ?? result.title;

    return (
        <motion.div
            className={`search-result-item ${isSelected ? 'selected' : ''}`}
            onClick={onClick}
            onMouseEnter={onMouseEnter}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.15 }}
            style={{
                padding: 'var(--space-3) var(--space-4)',
                cursor: 'pointer',
                borderRadius: 'var(--radius-md)',
                background: isSelected ? 'var(--bg-tertiary)' : 'transparent',
                transition: 'background-color 0.15s ease-out',
            }}
        >
            <div
                className="search-result-title"
                style={{ fontWeight: 500, color: 'var(--text-primary)', marginBottom: 'var(--space-1)' }}
            >
                {result.title}
            </div>
            <div
                className="search-result-path"
                style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 'var(--space-1)' }}
            >
                {result.path}
            </div>
            <div
                className="search-result-excerpt"
                style={{ fontSize: 'var(--text-sm)', color: 'var(--text-tertiary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            >
                {excerpt}
            </div>
        </motion.div>
    );
});

// ============================================
// Search Modal
// ============================================

interface SearchModalProps {
    onNavigate: NavigateFunction;
}

export function SearchModal({ onNavigate }: SearchModalProps): React.ReactElement | null {
    const [isOpen, openSearch, closeSearch] = useSearchState();
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SearchResult[]>([]);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const debouncedQuery = useDebounce(query, 200);

    // Keyboard shortcut to open search
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                openSearch();
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [openSearch]);

    // Focus input when modal opens
    useEffect(() => {
        if (isOpen && inputRef.current) {
            setTimeout(() => {
                inputRef.current?.focus();
                inputRef.current?.select();
            }, 50);
        }
    }, [isOpen]);

    // Perform search when debounced query changes
    useEffect(() => {
        if (!debouncedQuery.trim()) {
            setResults([]);
            return;
        }

        async function performSearch() {
            setIsLoading(true);
            try {
                const response = await api.search(debouncedQuery);
                setResults(response.results);
                setSelectedIndex(0);
            } catch (error) {
                console.error('Search error:', error);
                setResults([]);
            } finally {
                setIsLoading(false);
            }
        }

        performSearch();
    }, [debouncedQuery]);

    // Handle keyboard navigation
    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent) => {
            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    setSelectedIndex((prev) => (prev + 1) % Math.max(results.length, 1));
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    setSelectedIndex((prev) => (prev - 1 + results.length) % Math.max(results.length, 1));
                    break;
                case 'Enter':
                    e.preventDefault();
                    if (results[selectedIndex]) {
                        onNavigate(results[selectedIndex].path);
                        handleClose();
                    }
                    break;
                case 'Escape':
                    e.preventDefault();
                    handleClose();
                    break;
            }
        },
        [results, selectedIndex, onNavigate]
    );

    const handleClose = useCallback(() => {
        closeSearch();
        setQuery('');
        setResults([]);
        setSelectedIndex(0);
    }, [closeSearch]);

    const handleBackdropClick = useCallback(
        (e: React.MouseEvent) => {
            if (e.target === e.currentTarget) {
                handleClose();
            }
        },
        [handleClose]
    );

    const handleResultClick = useCallback(
        (result: SearchResult) => {
            onNavigate(result.path);
            handleClose();
        },
        [onNavigate, handleClose]
    );

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    id="search-modal"
                    className="search-modal"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    onClick={handleBackdropClick}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        zIndex: 1000,
                        display: 'flex',
                        alignItems: 'flex-start',
                        justifyContent: 'center',
                        paddingTop: '15vh',
                        background: 'var(--surface-overlay)',
                        backdropFilter: 'blur(4px)',
                    }}
                >
                    <motion.div
                        className="search-modal-content"
                        initial={{ scale: 0.95, y: -20 }}
                        animate={{ scale: 1, y: 0 }}
                        exit={{ scale: 0.95, y: -20 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                        style={{
                            width: '100%',
                            maxWidth: '640px',
                            background: 'var(--bg-elevated)',
                            border: '1px solid var(--border-primary)',
                            borderRadius: 'var(--radius-2xl)',
                            boxShadow: 'var(--shadow-2xl)',
                            overflow: 'hidden',
                        }}
                    >
                        {/* Search Input */}
                        <div
                            className="search-input-container"
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 'var(--space-3)',
                                padding: 'var(--space-4) var(--space-5)',
                                borderBottom: '1px solid var(--border-primary)',
                            }}
                        >
                            <svg
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                style={{ width: '20px', height: '20px', color: 'var(--text-muted)', flexShrink: 0 }}
                            >
                                <circle cx="11" cy="11" r="8" />
                                <path d="M21 21l-4.35-4.35" />
                            </svg>
                            <input
                                ref={inputRef}
                                id="search-input"
                                type="text"
                                className="search-input"
                                placeholder="Search documentation..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                onKeyDown={handleKeyDown}
                                autoComplete="off"
                                style={{
                                    flex: 1,
                                    border: 'none',
                                    background: 'none',
                                    fontSize: 'var(--text-lg)',
                                    color: 'var(--text-primary)',
                                    outline: 'none',
                                }}
                            />
                            <Kbd>ESC</Kbd>
                        </div>

                        {/* Search Results */}
                        <div
                            id="search-results"
                            className="search-results"
                            style={{ maxHeight: '400px', overflowY: 'auto', padding: 'var(--space-2)' }}
                        >
                            {!query.trim() && (
                                <div
                                    className="search-hint"
                                    style={{ padding: 'var(--space-8)', textAlign: 'center', color: 'var(--text-muted)' }}
                                >
                                    Type to search...
                                </div>
                            )}

                            {query.trim() && isLoading && (
                                <div
                                    style={{ padding: 'var(--space-8)', textAlign: 'center', color: 'var(--text-muted)' }}
                                >
                                    Searching...
                                </div>
                            )}

                            {query.trim() && !isLoading && results.length === 0 && (
                                <div
                                    className="search-no-results"
                                    style={{ padding: 'var(--space-8)', textAlign: 'center', color: 'var(--text-muted)' }}
                                >
                                    No results found for "<strong>{query}</strong>"
                                </div>
                            )}

                            {results.map((result, index) => (
                                <SearchResultItem
                                    key={result.path}
                                    result={result}
                                    isSelected={index === selectedIndex}
                                    onClick={() => handleResultClick(result)}
                                    onMouseEnter={() => setSelectedIndex(index)}
                                />
                            ))}
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}

export default SearchModal;
