/**
 * LoadingSpinner Component
 * 
 * Animated loading spinner with customizable size and color.
 */

import React from 'react';
import { motion } from 'framer-motion';

interface LoadingSpinnerProps {
    size?: number;
    color?: string;
    className?: string;
}

export function LoadingSpinner({
    size = 24,
    color = 'currentColor',
    className = '',
}: LoadingSpinnerProps): React.ReactElement {
    return (
        <motion.svg
            className={`loading-spinner ${className}`}
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        >
            <circle
                cx="12"
                cy="12"
                r="10"
                stroke={color}
                strokeWidth="3"
                strokeLinecap="round"
                strokeDasharray="60"
                strokeDashoffset="20"
                opacity={0.25}
            />
            <motion.circle
                cx="12"
                cy="12"
                r="10"
                stroke={color}
                strokeWidth="3"
                strokeLinecap="round"
                strokeDasharray="60"
                strokeDashoffset="45"
            />
        </motion.svg>
    );
}

export default LoadingSpinner;
