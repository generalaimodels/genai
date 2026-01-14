/**
 * RepoSearchCard Component
 * =========================
 * Premium glassmorphism URL input with animated submit button.
 * Validates GitHub URLs and triggers repository processing.
 */

import { useState, useCallback, type FormEvent, type ChangeEvent } from 'react';
import { motion } from 'framer-motion';

interface RepoSearchCardProps {
    onSubmit: (url: string) => Promise<void>;
    isLoading: boolean;
}

/** GitHub URL validation regex */
const GITHUB_URL_PATTERN = /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+(\.git)?$/;

export function RepoSearchCard({ onSubmit, isLoading }: RepoSearchCardProps) {
    const [url, setUrl] = useState('');
    const [isValid, setIsValid] = useState(true);

    const handleChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value;
        setUrl(value);
        setIsValid(value === '' || GITHUB_URL_PATTERN.test(value));
    }, []);

    const handleSubmit = useCallback(async (e: FormEvent) => {
        e.preventDefault();
        if (!url || !GITHUB_URL_PATTERN.test(url)) {
            setIsValid(false);
            return;
        }
        await onSubmit(url);
    }, [url, onSubmit]);

    return (
        <motion.div
            className="git-card repo-search"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
        >
            <div className="git-card__header">
                <svg className="git-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
                </svg>
                <h2 className="git-card__title">GitHub Repository</h2>
            </div>

            <div className="git-card__content">
                <form className="repo-search__form" onSubmit={handleSubmit}>
                    <div className="repo-search__input-wrapper">
                        <input
                            type="text"
                            className="repo-search__input"
                            placeholder="https://github.com/owner/repository.git"
                            value={url}
                            onChange={handleChange}
                            disabled={isLoading}
                            style={{
                                borderColor: !isValid ? 'var(--agion-color-error)' : undefined
                            }}
                        />
                        <svg className="repo-search__input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                        </svg>
                    </div>

                    <motion.button
                        type="submit"
                        className="repo-search__submit"
                        disabled={isLoading || !url}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        {isLoading ? (
                            <>
                                <svg className="repo-search__spinner" width="16" height="16" viewBox="0 0 24 24">
                                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" strokeDasharray="31.4" strokeDashoffset="10">
                                        <animateTransform attributeName="transform" type="rotate" dur="1s" from="0 12 12" to="360 12 12" repeatCount="indefinite" />
                                    </circle>
                                </svg>
                                Processing...
                            </>
                        ) : (
                            <>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="7 10 12 15 17 10" />
                                    <line x1="12" y1="15" x2="12" y2="3" />
                                </svg>
                                Clone & Index
                            </>
                        )}
                    </motion.button>
                </form>
            </div>
        </motion.div>
    );
}
