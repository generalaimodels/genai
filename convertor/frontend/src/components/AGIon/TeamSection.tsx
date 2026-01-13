/**
 * Team Section Component - AGIon Landing Page
 * 
 * FEATURES:
 * - Real team data integration (Hemanth profile)
 * - Grayscale-to-color hover effect on images
 * - Real social media links (GitHub, X, LinkedIn)
 */

import React from 'react';
import { Button, Icon } from './shared';
import './TeamSection.css';

interface TeamMember {
    name: string;
    role: string;
    bio: string;
    image?: string;
    links: {
        github?: string;
        githubAlt?: string;
        x?: string;
        linkedin?: string;
    };
}

const teamMembers: TeamMember[] = [
    {
        name: 'Hemanth K',
        role: 'Founder & Lead Engineer',
        bio: 'Building end-to-end AI systems with research rigor and production focus. Passionate about making frontier AI research accessible and actionable.',
        links: {
            github: 'https://github.com/generalaimodels',
            githubAlt: 'https://github.com/HemanthIITJ',
            x: 'https://x.com/Hemanth2022pee',
            linkedin: 'https://linkedin.com/in/hemanth-k-a88786215',
        },
    },
];

export const TeamSection: React.FC = () => {
    return (
        <section className="team-section agion-section" id="team">
            <div className="agion-container">
                <div className="team-header">
                    <h2 className="team-title">Superstar Team</h2>
                    <p className="team-description">
                        Building the future of AI systems with research excellence and engineering rigor.
                    </p>
                </div>

                <div className="team-grid">
                    {teamMembers.map((member) => (
                        <div key={member.name} className="team-card">
                            <div className="team-card-image">
                                {/* Placeholder gradient avatar */}
                                <div className="team-avatar">
                                    <svg width="120" height="120" viewBox="0 0 120 120">
                                        <defs>
                                            <linearGradient id="avatar-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                                <stop offset="0%" stopColor="#8B5CF6" />
                                                <stop offset="100%" stopColor="#EC4899" />
                                            </linearGradient>
                                        </defs>
                                        <circle cx="60" cy="60" r="60" fill="url(#avatar-gradient)" />
                                        <circle cx="60" cy="45" r="20" fill="white" opacity="0.9" />
                                        <circle cx="60" cy="85" r="30" fill="white" opacity="0.9" />
                                    </svg>
                                </div>
                            </div>

                            <h3 className="team-card-name">{member.name}</h3>
                            <p className="team-card-role agion-mono">{member.role}</p>
                            <p className="team-card-bio">{member.bio}</p>

                            <div className="team-card-links">
                                {member.links.github && (
                                    <a
                                        href={member.links.github}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="team-social-link"
                                        aria-label="GitHub"
                                    >
                                        <Icon name="github" size={20} />
                                    </a>
                                )}
                                {member.links.x && (
                                    <a
                                        href={member.links.x}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="team-social-link"
                                        aria-label="X (Twitter)"
                                    >
                                        <Icon name="x" size={20} />
                                    </a>
                                )}
                                {member.links.linkedin && (
                                    <a
                                        href={member.links.linkedin}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="team-social-link"
                                        aria-label="LinkedIn"
                                    >
                                        <Icon name="linkedin" size={20} />
                                    </a>
                                )}
                            </div>
                        </div>
                    ))}
                </div>

                <div className="team-cta">
                    <Button variant="primary" size="lg">
                        Work with us
                    </Button>
                </div>
            </div>
        </section>
    );
};
