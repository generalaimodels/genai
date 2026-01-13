/**
 * Navigation Bar Component - AGIon Landing Page
 * 
 * FEATURES:
 * - Sticky positioning with glassmorphism blur
 * - Desktop: Centered nav links, right-aligned CTAs
 * - Mobile: Hamburger menu with slide-over panel
 */

import React, { useState } from 'react';
import { Button, Icon } from './shared';
import './NavigationBar.css';

export const NavigationBar: React.FC = () => {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    const navLinks = [
        { label: 'About', href: '#about' },
        { label: 'Research', href: '#research' },
        { label: 'Knowledge Hub', href: '#knowledge' },
        { label: 'Platform', href: '#platform' },
        { label: 'Team', href: '#team' },
        { label: 'Community', href: '#community' },
        { label: 'Partners', href: '#partners' },
    ];

    return (
        <nav className="agion-nav">
            <div className="agion-container">
                <div className="nav-content">
                    {/* Left: Logo + Brand */}
                    <div className="nav-left">
                        <div className="nav-logo">
                            <img
                                src="/src/asserts/logopng.png"
                                alt="AEGIS AI Logo"
                                className="nav-logo-img"
                            />
                        </div>
                        <span className="nav-brand">AEGIS AI</span>
                    </div>

                    {/* Center: Desktop Links */}
                    <div className="nav-center agion-hidden-mobile">
                        {navLinks.map((link) => (
                            <a key={link.href} href={link.href} className="nav-link">
                                {link.label}
                            </a>
                        ))}
                    </div>

                    {/* Right: Actions */}
                    <div className="nav-right agion-hidden-mobile">
                        <Button variant="ghost" size="sm">
                            Sign in
                        </Button>
                        <Button variant="primary" size="sm">
                            Explore the Platform
                        </Button>
                    </div>

                    {/* Mobile: Hamburger */}
                    <button
                        className="nav-mobile-toggle agion-hidden-desktop"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                        aria-label="Toggle navigation menu"
                    >
                        <Icon name={mobileMenuOpen ? 'close' : 'menu'} size={24} />
                    </button>
                </div>
            </div>

            {/* Mobile Menu Panel */}
            <div className={`nav-mobile-menu ${mobileMenuOpen ? 'open' : ''}`}>
                <div className="nav-mobile-links">
                    {navLinks.map((link) => (
                        <a
                            key={link.href}
                            href={link.href}
                            className="nav-mobile-link"
                            onClick={() => setMobileMenuOpen(false)}
                        >
                            {link.label}
                        </a>
                    ))}
                </div>
                <div className="nav-mobile-actions">
                    <Button variant="ghost" size="md">
                        Sign in
                    </Button>
                    <Button variant="primary" size="md">
                        Explore the Platform
                    </Button>
                </div>
                <p className="nav-mobile-footer agion-mono">
                    Advanced Engineering for General Intelligence Systems.
                </p>
            </div>
        </nav>
    );
};
