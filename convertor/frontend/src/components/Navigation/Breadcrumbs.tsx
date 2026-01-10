/**
 * Breadcrumbs Component
 * 
 * Shows document hierarchy with clickable navigation.
 */

import React from 'react';
import { motion } from 'framer-motion';

interface BreadcrumbsProps {
    path: string;
    className?: string;
}

export function Breadcrumbs({ path, className = '' }: BreadcrumbsProps): React.ReactElement {
    const parts = path.split('/').filter(Boolean);

    if (parts.length === 0) return <></>;

    return (
        <nav className={`breadcrumbs ${className}`} aria-label="Breadcrumb">
            {parts.map((part, index) => {
                const href = '/' + parts.slice(0, index + 1).join('/');
                const isLast = index === parts.length - 1;

                return (
                    <React.Fragment key={index}>
                        {index > 0 && (
                            <span className="separator" aria-hidden="true">
                                /
                            </span>
                        )}
                        {isLast ? (
                            <span className="breadcrumb-current" aria-current="page">
                                {part.replace(/-/g, ' ').replace(/_/g, ' ')}
                            </span>
                        ) : (
                            <motion.a
                                href={href}
                                className="breadcrumb-link"
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                            >
                                {part.replace(/-/g, ' ').replace(/_/g, ' ')}
                            </motion.a>
                        )}
                    </React.Fragment>
                );
            })}
        </nav>
    );
}

export default Breadcrumbs;
