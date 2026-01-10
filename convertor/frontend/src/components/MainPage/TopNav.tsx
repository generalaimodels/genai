/**
 * TopNav Component
 * 
 * Glassmorphism top navigation bar with:
 * - Neural Frontiers Lab logo
 * - Research, Science, About, Blog dropdowns
 * - Premium blur effects
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface NavLink {
    id: string;
    label: string;
    hasDropdown?: boolean;
    items?: { label: string; href: string }[];
}

const navLinks: NavLink[] = [
    {
        id: 'research',
        label: 'Research',
        hasDropdown: true,
        items: [
            { label: 'Latest Papers', href: '#papers' },
            { label: 'Projects', href: '#projects' },
            { label: 'Collaborations', href: '#collaborations' }
        ]
    },
    {
        id: 'science',
        label: 'Science',
        hasDropdown: true,
        items: [
            { label: 'Neural Networks', href: '#neural' },
            { label: 'Machine Learning', href: '#ml' },
            { label: 'AI Safety', href: '#safety' }
        ]
    },
    { id: 'about', label: 'About', hasDropdown: false },
    { id: 'blog', label: 'Blog', hasDropdown: false }
];

export function TopNav(): React.ReactElement {
    const [activeDropdown, setActiveDropdown] = useState<string | null>(null);

    return (
        <motion.header
            className="topnav"
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
            <div className="topnav-inner">
                {/* Logo - Minimalist Premium DeepMind Style */}
                <div className="topnav-logo">
                    <div className="topnav-logo-icon">
                        <svg viewBox="0 0 32 32" fill="none">
                            <circle cx="16" cy="16" r="12" stroke="white" strokeWidth="2" strokeOpacity="0.9" />
                            <path d="M16 8V24" stroke="white" strokeWidth="2" strokeOpacity="0.9" />
                            <path d="M8 16H24" stroke="white" strokeWidth="2" strokeOpacity="0.9" />
                            <circle cx="16" cy="16" r="4" fill="white" />
                        </svg>
                    </div>
                    <span className="topnav-logo-text">
                        NEURAL FRONTIERS LAB
                    </span>
                </div>

                {/* Navigation Links */}
                <nav className="topnav-links">
                    {navLinks.map((link) => (
                        <div
                            key={link.id}
                            className="topnav-item"
                            onMouseEnter={() => link.hasDropdown && setActiveDropdown(link.id)}
                            onMouseLeave={() => setActiveDropdown(null)}
                        >
                            <button className="topnav-link">
                                {link.label}
                                {link.hasDropdown && (
                                    <motion.svg
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        className="dropdown-arrow"
                                        animate={{ rotate: activeDropdown === link.id ? 180 : 0 }}
                                    >
                                        <polyline points="6 9 12 15 18 9" />
                                    </motion.svg>
                                )}
                            </button>

                            <AnimatePresence>
                                {link.hasDropdown && activeDropdown === link.id && (
                                    <motion.div
                                        className="topnav-dropdown"
                                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        exit={{ opacity: 0, y: 10, scale: 1 }} // Fixed exit scale
                                        transition={{ duration: 0.2 }}
                                    >
                                        {link.items?.map((item) => (
                                            <a
                                                key={item.label}
                                                href={item.href}
                                                className="dropdown-item"
                                            >
                                                {item.label}
                                            </a>
                                        ))}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    ))}
                </nav>
            </div>
        </motion.header>
    );
}

export default TopNav;
