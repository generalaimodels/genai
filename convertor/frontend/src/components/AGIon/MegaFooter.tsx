/**
 * Mega Footer Component - AGIon Landing Page
 * 
 * LAYOUT: Newsletter row → 5-column grid → Bottom bar with social links
 * INTEGRATION: Real social media links for Hemanth
 */

import React, { useState } from 'react';
import { Button, Icon } from './shared';
import './MegaFooter.css';

export const MegaFooter: React.FC = () => {
    const [email, setEmail] = useState('');

    const handleNewsletterSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        console.log('Newsletter signup:', email);
        // TODO: Integrate with newsletter API
        setEmail('');
    };

    return (
        <footer className="mega-footer">
            <div className="agion-container">
                {/* Newsletter Row */}
                <div className="footer-newsletter">
                    <div className="newsletter-content">
                        <h3 className="newsletter-title">Stay Updated</h3>
                        <p className="newsletter-description">
                            Get research notes, system design patterns, and community updates.
                        </p>
                    </div>
                    <form className="newsletter-form" onSubmit={handleNewsletterSubmit}>
                        <input
                            type="email"
                            placeholder="your@email.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="newsletter-input"
                            required
                        />
                        <Button type="submit" variant="primary" size="md">
                            Subscribe
                        </Button>
                    </form>
                </div>

                {/* 5-Column Grid */}
                <div className="footer-grid">
                    {/* Product */}
                    <div className="footer-column">
                        <h4 className="footer-column-title">Product</h4>
                        <a href="#platform" className="footer-link">Platform</a>
                        <a href="#evals" className="footer-link">Evals Workbench</a>
                        <a href="#agents" className="footer-link">Agent Studio</a>
                        <a href="#knowledge-graph" className="footer-link">Knowledge Graph</a>
                    </div>

                    {/* Research */}
                    <div className="footer-column">
                        <h4 className="footer-column-title">Research</h4>
                        <a href="#notes" className="footer-link">Research Notes</a>
                        <a href="#publications" className="footer-link">Publications</a>
                        <a href="#safety" className="footer-link">Safety & Alignment</a>
                        <a href="#collaboration" className="footer-link">Collaborate</a>
                    </div>

                    {/* Knowledge */}
                    <div className="footer-column">
                        <h4 className="footer-column-title">Knowledge</h4>
                        <a href="#courses" className="footer-link">Courses</a>
                        <a href="#tutorials" className="footer-link">Tutorials</a>
                        <a href="#fundamentals" className="footer-link">Fundamentals</a>
                        <a href="#reading-paths" className="footer-link">Reading Paths</a>
                    </div>

                    {/* Company */}
                    <div className="footer-column">
                        <h4 className="footer-column-title">Company</h4>
                        <a href="#about" className="footer-link">About</a>
                        <a href="#team" className="footer-link">Team</a>
                        <a href="#careers" className="footer-link">Careers</a>
                        <a href="#blog" className="footer-link">Blog</a>
                    </div>

                    {/* Legal */}
                    <div className="footer-column">
                        <h4 className="footer-column-title">Legal</h4>
                        <a href="#terms" className="footer-link">Terms of Service</a>
                        <a href="#privacy" className="footer-link">Privacy Policy</a>
                        <a href="#security" className="footer-link">Security</a>
                    </div>
                </div>

                {/* Bottom Bar */}
                <div className="footer-bottom">
                    <p className="footer-copyright">© 2026 AEGIS AI. All rights reserved.</p>

                    <div className="footer-social">
                        <a
                            href="https://github.com/generalaimodels"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="social-link"
                            aria-label="GitHub"
                        >
                            <Icon name="github" size={20} />
                        </a>
                        <a
                            href="https://x.com/Hemanth2022pee"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="social-link"
                            aria-label="X (Twitter)"
                        >
                            <Icon name="x" size={20} />
                        </a>
                        <a
                            href="https://linkedin.com/in/hemanth-k-a88786215"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="social-link"
                            aria-label="LinkedIn"
                        >
                            <Icon name="linkedin" size={20} />
                        </a>
                    </div>

                    <p className="footer-microcopy agion-mono">
                        Build capability. Measure reliability. Earn trust.
                    </p>
                </div>
            </div>
        </footer>
    );
};
