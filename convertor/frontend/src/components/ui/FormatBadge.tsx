/**
 * FormatBadge Component
 * 
 * Displays document format with icon.
 * Showcases Phase 2 format converter support.
 */

import React from 'react';
import { motion } from 'framer-motion';

interface FormatBadgeProps {
    format: string;
    className?: string;
}

const FORMAT_CONFIG: Record<string, { label: string; icon: string; color: string }> = {
    md: { label: 'Markdown', icon: '📝', color: '#0969da' },
    rst: { label: 'ReStructuredText', icon: '📋', color: '#8B4513' },
    ipynb: { label: 'Jupyter Notebook', icon: '📓', color: '#F37726' },
    mdx: { label: 'MDX', icon: '⚡', color: '#1B1F24' },
    Rd: { label: 'R Documentation', icon: '📊', color: '#276DC3' },
};

export function FormatBadge({ format, className = '' }: FormatBadgeProps): React.ReactElement {
    const config = FORMAT_CONFIG[format] || {
        label: format.toUpperCase(),
        icon: '📄',
        color: '#64748b'
    };

    return (
        <motion.span
            className={`format-badge ${className}`}
            style={{
                borderColor: config.color,
                color: config.color,
            }}
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
            title={`${config.label} format`}
        >
            <span className="format-icon">{config.icon}</span>
            <span className="format-label">{config.label}</span>
        </motion.span>
    );
}

export default FormatBadge;
