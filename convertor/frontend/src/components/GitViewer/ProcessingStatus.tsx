/**
 * ProcessingStatus Component
 * ===========================
 * Real-time processing indicator with animated progress bar.
 * Shows cloning/processing status with file counts.
 */

import { motion, AnimatePresence } from 'framer-motion';
import type { RepoMetadata, JobStatus } from './types';

interface ProcessingStatusProps {
    status: RepoMetadata | null;
}

const STATUS_CONFIG: Record<JobStatus, { label: string; icon: string }> = {
    pending: { label: 'Pending', icon: '⏳' },
    cloning: { label: 'Cloning Repository', icon: '📥' },
    processing: { label: 'Indexing Files', icon: '⚙️' },
    completed: { label: 'Ready', icon: '✓' },
    failed: { label: 'Failed', icon: '✗' }
};

export function ProcessingStatus({ status }: ProcessingStatusProps) {
    if (!status) return null;

    const config = STATUS_CONFIG[status.status as JobStatus] || STATUS_CONFIG.pending;
    const progress = status.total_files > 0
        ? (status.processed_files / status.total_files) * 100
        : 0;
    const isActive = status.status === 'cloning' || status.status === 'processing';

    return (
        <AnimatePresence>
            <motion.div
                className="git-card processing-status"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
            >
                <div className="git-card__header">
                    <svg className="git-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 6v6l4 2" />
                    </svg>
                    <h2 className="git-card__title">Processing Status</h2>
                    <span className={`processing-status__badge processing-status__badge--${status.status}`}>
                        {config.icon} {config.label}
                    </span>
                </div>

                <div className="git-card__content">
                    <div className="processing-status__content">
                        {isActive && (
                            <div className="processing-status__spinner" />
                        )}

                        <div className="processing-status__info">
                            <h3 className="processing-status__title">{status.name}</h3>
                            <p className="processing-status__subtitle">
                                {status.status === 'cloning' && 'Downloading repository files...'}
                                {status.status === 'processing' && `Indexed ${status.processed_files.toLocaleString()} of ${status.total_files.toLocaleString()} files`}
                                {status.status === 'completed' && `${status.total_files.toLocaleString()} files ready`}
                                {status.status === 'failed' && 'An error occurred during processing'}
                            </p>

                            {(status.status === 'processing' || status.status === 'completed') && (
                                <div className="processing-status__progress">
                                    <motion.div
                                        className="processing-status__progress-bar"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${progress}%` }}
                                        transition={{ duration: 0.5, ease: 'easeOut' }}
                                    />
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </motion.div>
        </AnimatePresence>
    );
}
