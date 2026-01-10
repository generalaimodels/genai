/**
 * ResearchCard Component
 * 
 * Premium glassmorphism card for research topics with:
 * - Animated gradient borders
 * - Hover glow effects
 * - Icon animations
 * - Micro-interactions
 */

import React from 'react';
import { motion } from 'framer-motion';

export interface ResearchCardProps {
    category: string;
    title: string;
    icon: React.ReactNode;
    variant?: 'default' | 'featured' | 'highlight';
    delay?: number;
    onClick?: () => void;
    className?: string; // Support for Bento Grid classes
}

export function ResearchCard({
    category,
    title,
    icon,
    variant = 'default',
    delay = 0,
    onClick,
    className = ''
}: ResearchCardProps): React.ReactElement {
    return (
        <motion.article
            className={`research-card research-card--${variant} ${className}`}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
                duration: 0.6,
                delay: delay * 0.1,
                ease: [0.16, 1, 0.3, 1]
            }}
            whileHover={{
                y: -8,
                transition: { duration: 0.3 }
            }}
            onClick={onClick}
        >
            {/* Gradient border effect */}
            <div className="card-glow" />

            {/* Card content */}
            <div className="card-inner">
                <div className="card-header">
                    <span className="card-category">{category}</span>
                </div>

                <h3 className="card-title">{title}</h3>

                <motion.div
                    className="card-icon"
                    whileHover={{
                        scale: 1.1,
                        rotate: 5,
                        transition: { duration: 0.2 }
                    }}
                >
                    {icon}
                </motion.div>
            </div>

            {/* Corner accent */}
            <div className="card-accent" />
        </motion.article>
    );
}

// Icon components for research cards
export const GeminiIcon = (): React.ReactElement => (
    <svg viewBox="0 0 48 48" fill="none">
        <defs>
            <linearGradient id="geminiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#ec4899" />
            </linearGradient>
        </defs>
        <path
            d="M24 4C12.954 4 4 12.954 4 24s8.954 20 20 20 20-8.954 20-20S35.046 4 24 4z"
            stroke="url(#geminiGrad)"
            strokeWidth="2"
            fill="none"
        />
        <path
            d="M24 12c-6.627 0-12 5.373-12 12s5.373 12 12 12 12-5.373 12-12-5.373-12-12-12z"
            stroke="url(#geminiGrad)"
            strokeWidth="2"
            fill="none"
            opacity="0.6"
        />
        <circle cx="24" cy="24" r="4" fill="url(#geminiGrad)" />
    </svg>
);

export const VisualIcon = (): React.ReactElement => (
    <svg viewBox="0 0 48 48" fill="none">
        <defs>
            <linearGradient id="visualGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" />
                <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
        </defs>
        <ellipse cx="24" cy="24" rx="18" ry="12" stroke="url(#visualGrad)" strokeWidth="2" fill="none" />
        <circle cx="24" cy="24" r="6" stroke="url(#visualGrad)" strokeWidth="2" fill="none" />
        <circle cx="24" cy="24" r="2" fill="url(#visualGrad)" />
    </svg>
);

export const BCIIcon = (): React.ReactElement => (
    <svg viewBox="0 0 48 48" fill="none">
        <defs>
            <linearGradient id="bciGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f59e0b" />
                <stop offset="100%" stopColor="#ec4899" />
            </linearGradient>
        </defs>
        <path
            d="M24 8c-8.837 0-16 7.163-16 16v8c0 2.2 1.8 4 4 4h24c2.2 0 4-1.8 4-4v-8c0-8.837-7.163-16-16-16z"
            stroke="url(#bciGrad)"
            strokeWidth="2"
            fill="none"
        />
        <circle cx="18" cy="22" r="2" fill="url(#bciGrad)" />
        <circle cx="30" cy="22" r="2" fill="url(#bciGrad)" />
        <path d="M18 30c2 2 4 3 6 3s4-1 6-3" stroke="url(#bciGrad)" strokeWidth="2" fill="none" />
    </svg>
);

export const TeamIcon = (): React.ReactElement => (
    <svg viewBox="0 0 48 48" fill="none">
        <defs>
            <linearGradient id="teamGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
        </defs>
        <circle cx="24" cy="16" r="6" stroke="url(#teamGrad)" strokeWidth="2" fill="none" />
        <path
            d="M12 36c0-6.627 5.373-12 12-12s12 5.373 12 12"
            stroke="url(#teamGrad)"
            strokeWidth="2"
            fill="none"
        />
        <circle cx="10" cy="20" r="4" stroke="url(#teamGrad)" strokeWidth="1.5" fill="none" opacity="0.6" />
        <circle cx="38" cy="20" r="4" stroke="url(#teamGrad)" strokeWidth="1.5" fill="none" opacity="0.6" />
    </svg>
);

export const NeuralIcon = (): React.ReactElement => (
    <svg viewBox="0 0 48 48" fill="none">
        <defs>
            <linearGradient id="neuralGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#c084fc" />
                <stop offset="100%" stopColor="#f472b6" />
            </linearGradient>
        </defs>
        <circle cx="24" cy="12" r="4" fill="url(#neuralGrad)" />
        <circle cx="12" cy="28" r="4" fill="url(#neuralGrad)" opacity="0.8" />
        <circle cx="36" cy="28" r="4" fill="url(#neuralGrad)" opacity="0.8" />
        <circle cx="24" cy="38" r="3" fill="url(#neuralGrad)" opacity="0.6" />
        <line x1="24" y1="12" x2="12" y2="28" stroke="url(#neuralGrad)" strokeWidth="2" opacity="0.5" />
        <line x1="24" y1="12" x2="36" y2="28" stroke="url(#neuralGrad)" strokeWidth="2" opacity="0.5" />
        <line x1="12" y1="28" x2="24" y2="38" stroke="url(#neuralGrad)" strokeWidth="2" opacity="0.4" />
        <line x1="36" y1="28" x2="24" y2="38" stroke="url(#neuralGrad)" strokeWidth="2" opacity="0.4" />
        <line x1="12" y1="28" x2="36" y2="28" stroke="url(#neuralGrad)" strokeWidth="1.5" opacity="0.3" />
    </svg>
);

export const StarIcon = (): React.ReactElement => (
    <svg viewBox="0 0 24 24" fill="none">
        <defs>
            <linearGradient id="starGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#fbbf24" />
                <stop offset="100%" stopColor="#f59e0b" />
            </linearGradient>
        </defs>
        <path
            d="M12 2l2.4 7.4h7.6l-6.2 4.5 2.4 7.4L12 16.8l-6.2 4.5 2.4-7.4L2 9.4h7.6z"
            fill="url(#starGrad)"
        />
    </svg>
);

export default ResearchCard;
