/**
 * Hero Section Component - AGIon Landing Page
 * 
 * LAYOUT: Asymmetric 60/40 grid, min-height 90vh
 * LEFT ZONE: Eyebrow → H1 Headline → Subheadline → Action Cluster → Proof Points
 * RIGHT ZONE: Glass companion card with interactive list
 */

import React from 'react';
import { Button, Icon } from './shared';
import './HeroSection.css';

export const HeroSection: React.FC = () => {
    return (
        <section className="hero-section">
            <div className="agion-container">
                <div className="hero-grid">
                    {/* LEFT ZONE */}
                    <div className="hero-left">
                        <p className="hero-eyebrow agion-eyebrow">
                            ADVANCED ENGINEERING • GENERAL INTELLIGENCE SYSTEMS • OPEN KNOWLEDGE
                        </p>

                        <h1 className="hero-headline">
                            Advanced Engineering for{' '}
                            <span className="agion-gradient-text">General Intelligence</span> Systems
                        </h1>

                        <p className="hero-subheadline">
                            AEGIS AI is an end-to-end AI knowledge platform combining frontier research,
                            production systems, and open educational resources for AI engineers.
                        </p>

                        <div className="hero-actions">
                            <Button variant="primary" size="lg">
                                Explore the Platform
                            </Button>
                            <Button variant="secondary" size="lg">
                                Read the Vision
                            </Button>
                            <a href="#research" className="hero-link agion-underline-animated">
                                View Research Notes
                                <Icon name="arrow-right" size={16} className="hero-link-icon" />
                            </a>
                        </div>

                        <div className="hero-proof-points">
                            <div className="proof-point">
                                <Icon name="research" size={20} className="proof-icon" />
                                <span>Research, translated</span>
                            </div>
                            <div className="proof-point">
                                <Icon name="system" size={20} className="proof-icon" />
                                <span>Systems, not demos</span>
                            </div>
                            <div className="proof-point">
                                <Icon name="community" size={20} className="proof-icon" />
                                <span>Community of builders</span>
                            </div>
                        </div>
                    </div>

                    {/* RIGHT ZONE */}
                    <div className="hero-right">
                        <div className="hero-companion-card agion-glass">
                            <h3 className="companion-title">Start here</h3>

                            <div className="companion-list">
                                <div className="companion-item">
                                    <Icon name="blueprint" size={20} className="companion-icon" />
                                    <span>Build an AI Agent (Blueprint)</span>
                                </div>
                                <div className="companion-item">
                                    <Icon name="book" size={20} className="companion-icon" />
                                    <span>LLM Systems Fundamentals (Course)</span>
                                </div>
                                <div className="companion-item">
                                    <Icon name="check" size={20} className="companion-icon" />
                                    <span>Eval-First Deployment (Checklist)</span>
                                </div>
                                <div className="companion-item">
                                    <Icon name="shield" size={20} className="companion-icon" />
                                    <span>Safety & Alignment Basics</span>
                                </div>
                            </div>

                            <p className="companion-footer agion-mono">
                                No fluff. Clear prerequisites. Production-ready patterns.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};
