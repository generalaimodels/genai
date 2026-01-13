/**
 * AGIonLanding Component - Main Landing Page Assembly
 * 
 * ARCHITECTURE:
 * - Assembles all section components into complete landing page
 * - Imports all design system CSS
 * - Premium + Classic color scheme
 */

import React, { useEffect } from 'react';
import { NavigationBar } from './NavigationBar';
import { HeroSection } from './HeroSection';
import { TeamSection } from './TeamSection';
import { MegaFooter } from './MegaFooter';

// Import design system CSS
import '../../styles/aegis-tokens.css';
import '../../styles/aegis-grid.css';
import '../../styles/aegis-typography.css';
import '../../styles/aegis-animations.css';

import './AGIonLanding.css';

export const AGIonLanding: React.FC = () => {
    useEffect(() => {
        // Scroll-to-top on mount
        window.scrollTo(0, 0);

        // Smooth scroll for anchor links
        const handleAnchorClick = (e: MouseEvent) => {
            const target = e.target as HTMLAnchorElement;
            if (target.tagName === 'A' && target.hash) {
                const element = document.querySelector(target.hash);
                if (element) {
                    e.preventDefault();
                    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        };

        document.addEventListener('click', handleAnchorClick);
        return () => {
            document.removeEventListener('click', handleAnchorClick);
        };
    }, []);

    return (
        <div className="agion-landing">
            <NavigationBar />
            <main>
                <HeroSection />

                {/* About / Vision Section */}
                <section className="about-section agion-section" id="about">
                    <div className="agion-container">
                        <div className="section-header">
                            <h2 className="section-title">Why AEGIS AI exists</h2>
                            <p className="section-description agion-lead">
                                Building the bridge between frontier AI research and production systems.
                            </p>
                        </div>
                        <div className="agion-grid-3-col">
                            <div className="about-card">
                                <h3>Research is fragmented</h3>
                                <p>
                                    Cutting-edge research lives in papers, scattered across arXiv, conference proceedings,
                                    and lab blogs. Engineers waste weeks decoding opaque methodologies.
                                </p>
                            </div>
                            <div className="about-card">
                                <h3>Production is unforgiving</h3>
                                <p>
                                    Latency spikes. Prompt injections. Model drift. Production AI demands battle-tested
                                    patterns, not research demos.
                                </p>
                            </div>
                            <div className="about-card">
                                <h3>Knowledge isn't end-to-end</h3>
                                <p>
                                    Most resources cover either theory or implementation. AEGIS AI bridges both: research
                                    translated into production-ready systems.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Team Section */}
                <TeamSection />

                {/* Community Section */}
                <section className="community-section agion-section" id="community">
                    <div className="agion-container">
                        <div className="section-header">
                            <h2 className="section-title">Join the Community</h2>
                            <p className="section-description agion-lead">
                                Contributors, builders, and partners building the future of AI systems.
                            </p>
                        </div>
                        <div className="agion-grid-3-col">
                            <div className="community-card">
                                <h3>Contributors</h3>
                                <p>Join our open-source contributors and shape the future of AI knowledge.</p>
                            </div>
                            <div className="community-card">
                                <h3>Builders</h3>
                                <p>Connect with fellow AI engineers building production systems.</p>
                            </div>
                            <div className="community-card">
                                <h3>Partners</h3>
                                <p>Academic labs and enterprises pushing the frontier of AI research.</p>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
            <MegaFooter />
        </div>
    );
};
