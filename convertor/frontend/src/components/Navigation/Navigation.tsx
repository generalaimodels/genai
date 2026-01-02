/**
 * Navigation Component
 * 
 * Sidebar navigation tree with collapsible folders.
 * Features smooth expand/collapse animations.
 */

import React, { useEffect, useCallback, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '@/api/client';
import { useApp, useCurrentPath } from '@/context';
import { LoadingSpinner } from '@/components/ui';
import type { NavigationNode, NavigateFunction } from '@/types';

// ============================================
// NavItem Component
// ============================================

interface NavItemProps {
    node: NavigationNode;
    onNavigate: NavigateFunction;
}

const NavItem = memo(function NavItem({ node, onNavigate }: NavItemProps): React.ReactElement {
    const currentPath = useCurrentPath();
    const isActive = node.path === currentPath;

    const handleClick = useCallback(
        (e: React.MouseEvent) => {
            e.preventDefault();
            if (node.path) {
                onNavigate(node.path);
            }
        },
        [node.path, onNavigate]
    );

    return (
        <motion.a
            className={`nav-item ${isActive ? 'active' : ''}`}
            href={`#${node.path}`}
            data-path={node.path}
            onClick={handleClick}
            whileHover={{ x: 2 }}
            transition={{ duration: 0.15 }}
        >
            {node.name}
        </motion.a>
    );
});

// ============================================
// NavGroup Component
// ============================================

interface NavGroupProps {
    node: NavigationNode;
    onNavigate: NavigateFunction;
    defaultExpanded?: boolean;
}

const NavGroup = memo(function NavGroup({
    node,
    onNavigate,
    defaultExpanded = true,
}: NavGroupProps): React.ReactElement {
    const [isExpanded, setIsExpanded] = React.useState(defaultExpanded);
    const currentPath = useCurrentPath();

    // Auto-expand if current path is within this group
    useEffect(() => {
        if (currentPath && node.children.some((child) => child.path === currentPath)) {
            setIsExpanded(true);
        }
    }, [currentPath, node.children]);

    const toggleExpand = useCallback(() => {
        setIsExpanded((prev) => !prev);
    }, []);

    return (
        <div className={`nav-group ${isExpanded ? '' : 'collapsed'}`}>
            <motion.button
                className="nav-group-title"
                onClick={toggleExpand}
                whileHover={{ backgroundColor: 'var(--bg-tertiary)' }}
                transition={{ duration: 0.15 }}
            >
                <motion.svg
                    className="nav-group-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    animate={{ rotate: isExpanded ? 90 : 0 }}
                    transition={{ duration: 0.2, ease: 'easeOut' }}
                >
                    <path d="M9 18l6-6-6-6" />
                </motion.svg>
                {node.name}
            </motion.button>

            <AnimatePresence initial={false}>
                {isExpanded && (
                    <motion.div
                        className="nav-group-items"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.25, ease: 'easeOut' }}
                        style={{ overflow: 'hidden' }}
                    >
                        {node.children.map((child) =>
                            child.is_directory && child.children.length > 0 ? (
                                <NavGroup key={child.name} node={child} onNavigate={onNavigate} />
                            ) : child.path ? (
                                <NavItem key={child.path} node={child} onNavigate={onNavigate} />
                            ) : null
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
});

// ============================================
// Navigation Component
// ============================================

export function Navigation(): React.ReactElement {
    const { navigation, setNavigation, navigateTo } = useApp();
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    // Fetch navigation tree on mount
    useEffect(() => {
        async function loadNavigation() {
            try {
                const nav = await api.getNavigation();
                setNavigation(nav);
                setError(null);
            } catch (err) {
                console.error('Failed to load navigation:', err);
                setError('Failed to load navigation');
            } finally {
                setLoading(false);
            }
        }

        loadNavigation();
    }, [setNavigation]);

    if (loading) {
        return (
            <nav className="navigation">
                <div className="nav-loading" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 'var(--space-8)' }}>
                    <LoadingSpinner size={24} />
                </div>
            </nav>
        );
    }

    if (error) {
        return (
            <nav className="navigation">
                <div className="nav-error" style={{ padding: 'var(--space-4)', color: 'var(--color-error-500)', textAlign: 'center' }}>
                    {error}
                    <button
                        onClick={() => window.location.reload()}
                        style={{ display: 'block', margin: 'var(--space-2) auto', padding: 'var(--space-2) var(--space-4)', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}
                    >
                        Retry
                    </button>
                </div>
            </nav>
        );
    }

    if (!navigation) {
        return <nav className="navigation" />;
    }

    return (
        <nav id="navigation" className="navigation">
            {navigation.children.map((node) =>
                node.is_directory && node.children.length > 0 ? (
                    <NavGroup key={node.name} node={node} onNavigate={navigateTo} />
                ) : node.path ? (
                    <NavItem key={node.path} node={node} onNavigate={navigateTo} />
                ) : null
            )}
        </nav>
    );
}

export default Navigation;
