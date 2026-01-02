/**
 * Brain3D Component (CSS Version)
 * 
 * Replaces the unstable Three.js implementation with a robust, high-performance
 * CSS/SVG holographic animation. This ensures 100% stability while maintaining
 * the premium "Neural" aesthetic.
 */

import React from 'react';

export function Brain3D({ className = '' }: { className?: string }): React.ReactElement {
    return (
        <div className={`brain-container ${className}`} style={{
            width: '100%',
            height: '450px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            perspective: '1000px'
        }}>
            {/* Core Neural Sphere */}
            <div className="neural-sphere">
                {/* Orbital Rings */}
                <div className="ring ring-1"></div>
                <div className="ring ring-2"></div>
                <div className="ring ring-3"></div>

                {/* Central Core */}
                <div className="core-glow"></div>

                {/* Floating Particles (CSS) */}
                <div className="particle p1"></div>
                <div className="particle p2"></div>
                <div className="particle p3"></div>
            </div>

            <style>{`
                .neural-sphere {
                    position: relative;
                    width: 200px;
                    height: 200px;
                    transform-style: preserve-3d;
                    animation: float 6s ease-in-out infinite;
                }

                .ring {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    border: 2px solid rgba(139, 92, 246, 0.3);
                    border-radius: 50%;
                    box-shadow: 0 0 20px rgba(139, 92, 246, 0.2);
                }

                .ring-1 {
                    animation: rotate1 10s linear infinite;
                    border-top-color: #ec4899;
                    border-bottom-color: transparent;
                }

                .ring-2 {
                    width: 80%;
                    height: 80%;
                    top: 10%;
                    left: 10%;
                    animation: rotate2 15s linear infinite reverse;
                    border: 2px solid rgba(6, 182, 212, 0.3);
                    border-left-color: #06b6d4;
                    border-right-color: transparent;
                }

                .ring-3 {
                    width: 120%;
                    height: 120%;
                    top: -10%;
                    left: -10%;
                    animation: rotate3 20s linear infinite;
                    border: 1px dashed rgba(255, 255, 255, 0.2);
                }

                .core-glow {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 60px;
                    height: 60px;
                    background: radial-gradient(circle, #8b5cf6 0%, transparent 70%);
                    border-radius: 50%;
                    box-shadow: 0 0 40px 10px rgba(139, 92, 246, 0.4);
                    animation: pulse 3s ease-in-out infinite;
                }

                .particle {
                    position: absolute;
                    width: 6px;
                    height: 6px;
                    background: white;
                    border-radius: 50%;
                    box-shadow: 0 0 10px white;
                }
                
                .p1 { top: 20%; left: 20%; animation: orbit1 4s linear infinite; }
                .p2 { bottom: 30%; right: 20%; animation: orbit2 6s linear infinite; }
                .p3 { top: 50%; right: 10%; animation: orbit3 8s linear infinite; }

                @keyframes rotate1 {
                    0% { transform: rotateX(0deg) rotateY(0deg); }
                    100% { transform: rotateX(360deg) rotateY(360deg); }
                }
                @keyframes rotate2 {
                    0% { transform: rotateX(0deg) rotateY(0deg); }
                    100% { transform: rotateX(360deg) rotateY(-360deg); }
                }
                @keyframes rotate3 {
                    0% { transform: rotateZ(0deg); }
                    100% { transform: rotateZ(360deg); }
                }
                @keyframes float {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-20px); }
                }
                @keyframes pulse {
                    0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.8; }
                    50% { transform: translate(-50%, -50%) scale(1.2); opacity: 1; }
                }
                @keyframes orbit1 {
                    0% { transform: rotate(0deg) translateX(80px) rotate(0deg); }
                    100% { transform: rotate(360deg) translateX(80px) rotate(-360deg); }
                }
            `}</style>
        </div>
    );
}

export default Brain3D;
