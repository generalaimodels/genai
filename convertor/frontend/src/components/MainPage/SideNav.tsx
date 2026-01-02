/**
 * SideNav Component
 * 
 * Glassmorphism sidebar navigation with:
 * - Neural Frontiers Lab logo
 * - Navigation links (Home, Advancements, Publications, Team, Resources)
 * - Social links footer
 * - Premium hover animations
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface NavItem {
    id: string;
    label: string;
    icon: React.ReactNode;
    active?: boolean;
    href?: string; // Optional href for links
}

const navItems: NavItem[] = [
    {
        id: 'home',
        label: 'Home',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
                <polyline points="9,22 9,12 15,12 15,22" />
            </svg>
        ),
        active: true
    },
    {
        id: 'advancements',
        label: 'Advancements',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26" />
            </svg>
        )
    },
    {
        id: 'publications',
        label: 'Publications',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 19.5A2.5 2.5 0 016.5 17H20" />
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" />
            </svg>
        )
    },
    {
        id: 'team',
        label: 'Team',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 00-3-3.87" />
                <path d="M16 3.13a4 4 0 010 7.75" />
            </svg>
        )
    },
    {
        id: 'resources',
        label: 'Resources',
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="7" />
                <rect x="14" y="3" width="7" height="7" />
                <rect x="14" y="14" width="7" height="7" />
                <rect x="3" y="14" width="7" height="7" />
            </svg>
        ),
        href: '#/docs' // Link to documentation page
    }
];

interface SideNavProps {
    isCollapsed: boolean;
    toggleSidebar: () => void;
}

export function SideNav({ isCollapsed, toggleSidebar }: SideNavProps): React.ReactElement {
    const [activeId, setActiveId] = useState('home');

    return (
        <motion.aside
            className="sidebar"
            initial={false}
            animate={{ width: isCollapsed ? 72 : 250 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
            {/* Logo */}
            <div className="sidenav-header">
                <motion.div
                    className="sidenav-logo"
                    animate={{ scale: isCollapsed ? 0.9 : 1 }}
                >
                    <div className="logo-icon">
                        <svg viewBox="0 0 40 40" fill="none">
                            <defs>
                                <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#8b5cf6" />
                                    <stop offset="50%" stopColor="#c084fc" />
                                    <stop offset="100%" stopColor="#ec4899" />
                                </linearGradient>
                            </defs>
                            <circle cx="20" cy="20" r="18" fill="url(#logoGrad)" opacity="0.2" />
                            <circle cx="20" cy="14" r="4" fill="url(#logoGrad)" />
                            <circle cx="12" cy="24" r="3" fill="url(#logoGrad)" opacity="0.8" />
                            <circle cx="28" cy="24" r="3" fill="url(#logoGrad)" opacity="0.8" />
                            <circle cx="20" cy="30" r="2.5" fill="url(#logoGrad)" opacity="0.6" />
                            <line x1="20" y1="14" x2="12" y2="24" stroke="url(#logoGrad)" strokeWidth="1.5" opacity="0.6" />
                            <line x1="20" y1="14" x2="28" y2="24" stroke="url(#logoGrad)" strokeWidth="1.5" opacity="0.6" />
                            <line x1="12" y1="24" x2="20" y2="30" stroke="url(#logoGrad)" strokeWidth="1.5" opacity="0.5" />
                            <line x1="28" y1="24" x2="20" y2="30" stroke="url(#logoGrad)" strokeWidth="1.5" opacity="0.5" />
                        </svg>
                    </div>
                    <AnimatePresence>
                        {!isCollapsed && (
                            <motion.div
                                className="logo-text"
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -10 }}
                                transition={{ duration: 0.2 }}
                            >
                                <span className="logo-neural">NEURAL</span>
                                <span className="logo-frontiers">FRONTIERS</span>
                                <span className="logo-lab">LAB</span>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.div>
            </div>

            {/* Navigation */}
            <nav className="sidenav-nav">
                {navItems.map((item) => {
                    const Component = item.href ? 'a' : 'button';
                    const extraProps = item.href ? { href: item.href } : { onClick: () => setActiveId(item.id) };

                    return (
                        <motion.div
                            key={item.id}
                            style={{ position: 'relative' }}
                            whileHover={{ x: isCollapsed ? 0 : 4 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            <Component
                                className={`sidenav-item ${activeId === item.id ? 'active' : ''} ${isCollapsed ? 'collapsed' : ''}`}
                                {...extraProps}
                            >
                                <span className="sidenav-icon">{item.icon}</span>

                                <AnimatePresence>
                                    {!isCollapsed ? (
                                        <motion.span
                                            className="sidenav-label"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            exit={{ opacity: 0 }}
                                        >
                                            {item.label}
                                        </motion.span>
                                    ) : (
                                        /* Tooltip for collapsed state */
                                        <motion.div
                                            className="sidenav-tooltip"
                                            initial={{ opacity: 0, x: 10, scale: 0.9 }}
                                            whileHover={{ opacity: 1, x: 0, scale: 1 }}
                                            transition={{ duration: 0.2 }}
                                        >
                                            {item.label}
                                        </motion.div>
                                    )}
                                </AnimatePresence>

                                {activeId === item.id && (
                                    <motion.div
                                        className="sidenav-indicator"
                                        layoutId="activeIndicator"
                                        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                                    />
                                )}
                            </Component>
                        </motion.div>
                    );
                })}
            </nav>



            {/* Collapse Toggle */}
            <button
                className="sidenav-toggle"
                onClick={toggleSidebar}
                aria-label="Toggle sidebar"
            >
                <motion.svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    animate={{ rotate: isCollapsed ? 180 : 0 }}
                >
                    <polyline points="15 18 9 12 15 6" />
                </motion.svg>
            </button>
        </motion.aside>
    );
}

export default SideNav;
