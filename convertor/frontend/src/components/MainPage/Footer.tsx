/**
 * Footer Component
 * 
 * Premium professional footer with:
 * - Glassmorphism design
 * - Social links
 * - Navigation links
 * - Copyright information
 * - Smooth animations
 */

import React from 'react';
import { motion } from 'framer-motion';

export function Footer(): React.ReactElement {
    return (
        <motion.footer
            className="main-footer"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.2 }}
        >
            <div className="footer-content">
                {/* Footer Grid */}
                <div className="footer-grid">
                    {/* Brand Section */}
                    <div className="footer-section footer-brand">
                        <div className="footer-logo">
                            <svg viewBox="0 0 40 40" fill="none" className="footer-logo-svg">
                                <defs>
                                    <linearGradient id="footerLogoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#8b5cf6" />
                                        <stop offset="50%" stopColor="#c084fc" />
                                        <stop offset="100%" stopColor="#ec4899" />
                                    </linearGradient>
                                </defs>
                                <circle cx="20" cy="20" r="18" fill="url(#footerLogoGrad)" opacity="0.2" />
                                <circle cx="20" cy="14" r="4" fill="url(#footerLogoGrad)" />
                                <circle cx="12" cy="24" r="3" fill="url(#footerLogoGrad)" opacity="0.8" />
                                <circle cx="28" cy="24" r="3" fill="url(#footerLogoGrad)" opacity="0.8" />
                                <circle cx="20" cy="30" r="2.5" fill="url(#footerLogoGrad)" opacity="0.6" />
                                <line x1="20" y1="14" x2="12" y2="24" stroke="url(#footerLogoGrad)" strokeWidth="1.5" opacity="0.6" />
                                <line x1="20" y1="14" x2="28" y2="24" stroke="url(#footerLogoGrad)" strokeWidth="1.5" opacity="0.6" />
                                <line x1="12" y1="24" x2="20" y2="30" stroke="url(#footerLogoGrad)" strokeWidth="1.5" opacity="0.5" />
                                <line x1="28" y1="24" x2="20" y2="30" stroke="url(#footerLogoGrad)" strokeWidth="1.5" opacity="0.5" />
                            </svg>
                            <div className="footer-logo-text">
                                <span className="footer-neural">NEURAL</span>
                                <span className="footer-frontiers">FRONTIERS</span>
                            </div>
                        </div>
                        <p className="footer-tagline">
                            Pioneering research on the path to AGI
                        </p>
                    </div>

                    {/* Quick Links */}
                    <div className="footer-section">
                        <h3 className="footer-heading">Quick Links</h3>
                        <ul className="footer-links">
                            <li><a href="#home">Home</a></li>
                            <li><a href="#advancements">Advancements</a></li>
                            <li><a href="#publications">Publications</a></li>
                            <li><a href="#team">Team</a></li>
                        </ul>
                    </div>

                    {/* Resources */}
                    <div className="footer-section">
                        <h3 className="footer-heading">Resources</h3>
                        <ul className="footer-links">
                            <li><a href="#docs">Documentation</a></li>
                            <li><a href="#api">API Reference</a></li>
                            <li><a href="#blog">Blog</a></li>
                            <li><a href="#research">Research Papers</a></li>
                        </ul>
                    </div>

                    {/* Legal & Contact */}
                    <div className="footer-section">
                        <h3 className="footer-heading">Legal</h3>
                        <ul className="footer-links">
                            <li><a href="#privacy">Privacy Policy</a></li>
                            <li><a href="#terms">Terms of Service</a></li>
                            <li><a href="#contact">Contact Us</a></li>
                            <li><a href="#careers">Careers</a></li>
                        </ul>
                    </div>
                </div>

                {/* Divider */}
                <div className="footer-divider" />

                {/* Bottom Section */}
                <div className="footer-bottom">
                    <p className="footer-copyright">
                        © {new Date().getFullYear()} Neural Frontiers Lab. All rights reserved.
                    </p>

                    {/* Social Links */}
                    <div className="footer-social">
                        <a href="https://twitter.com" className="footer-social-link" aria-label="Twitter" target="_blank" rel="noopener noreferrer">
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                            </svg>
                        </a>
                        <a href="https://github.com" className="footer-social-link" aria-label="GitHub" target="_blank" rel="noopener noreferrer">
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
                            </svg>
                        </a>
                        <a href="https://linkedin.com" className="footer-social-link" aria-label="LinkedIn" target="_blank" rel="noopener noreferrer">
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                            </svg>
                        </a>
                        <a href="https://youtube.com" className="footer-social-link" aria-label="YouTube" target="_blank" rel="noopener noreferrer">
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
                            </svg>
                        </a>
                    </div>
                </div>
            </div>
        </motion.footer>
    );
}

export default Footer;
