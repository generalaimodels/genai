/**
 * NeuralBackground Component
 * 
 * High-performance animated neural network particle system
 * Uses canvas for 60fps smooth animations with requestAnimationFrame
 * Features: Floating particles, connecting lines, subtle glow effects
 */

import React, { useRef, useEffect, useCallback } from 'react';

interface Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    radius: number;
    opacity: number;
    hue: number;
}

interface NeuralBackgroundProps {
    particleCount?: number;
    connectionDistance?: number;
    speed?: number;
}

export function NeuralBackground({
    particleCount = 80,
    connectionDistance = 150,
    speed = 0.3
}: NeuralBackgroundProps): React.ReactElement {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const particlesRef = useRef<Particle[]>([]);
    const animationRef = useRef<number>(0);
    const mouseRef = useRef({ x: -1000, y: -1000 });

    // Initialize particles with neural-like properties
    const initParticles = useCallback((width: number, height: number) => {
        const particles: Particle[] = [];
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * speed,
                vy: (Math.random() - 0.5) * speed,
                radius: Math.random() * 2.5 + 1,
                opacity: Math.random() * 0.5 + 0.3,
                hue: Math.random() * 60 + 250 // Purple to cyan range
            });
        }
        particlesRef.current = particles;
    }, [particleCount, speed]);

    // Animation loop with requestAnimationFrame for 60fps
    const animate = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const { width, height } = canvas;
        const particles = particlesRef.current;
        const mouse = mouseRef.current;

        // Clear with fade effect for trail
        ctx.fillStyle = 'rgba(10, 5, 20, 0.15)';
        ctx.fillRect(0, 0, width, height);

        // Update and draw particles
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];

            // Update position
            p.x += p.vx;
            p.y += p.vy;

            // Bounce off edges smoothly
            if (p.x < 0 || p.x > width) p.vx *= -1;
            if (p.y < 0 || p.y > height) p.vy *= -1;

            // Mouse interaction - particles attracted to cursor
            const dx = mouse.x - p.x;
            const dy = mouse.y - p.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 200) {
                const force = (200 - dist) / 200;
                p.vx += dx * force * 0.0003;
                p.vy += dy * force * 0.0003;
            }

            // Speed limit
            const maxSpeed = speed * 2;
            const currentSpeed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
            if (currentSpeed > maxSpeed) {
                p.vx = (p.vx / currentSpeed) * maxSpeed;
                p.vy = (p.vy / currentSpeed) * maxSpeed;
            }

            // Draw glow
            const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 4);
            gradient.addColorStop(0, `hsla(${p.hue}, 80%, 70%, ${p.opacity})`);
            gradient.addColorStop(0.5, `hsla(${p.hue}, 80%, 50%, ${p.opacity * 0.3})`);
            gradient.addColorStop(1, 'transparent');

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius * 4, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Draw core particle
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${p.hue}, 90%, 80%, ${p.opacity})`;
            ctx.fill();
        }

        // Draw connections between nearby particles
        ctx.lineWidth = 0.5;
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const p1 = particles[i];
                const p2 = particles[j];
                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < connectionDistance) {
                    const opacity = (1 - dist / connectionDistance) * 0.4;
                    const gradient = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
                    gradient.addColorStop(0, `hsla(${p1.hue}, 70%, 60%, ${opacity})`);
                    gradient.addColorStop(1, `hsla(${p2.hue}, 70%, 60%, ${opacity})`);

                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = gradient;
                    ctx.stroke();
                }
            }
        }

        animationRef.current = requestAnimationFrame(animate);
    }, [connectionDistance, speed]);

    // Handle resize with debounce
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const handleResize = () => {
            const dpr = Math.min(window.devicePixelRatio, 2);
            canvas.width = window.innerWidth * dpr;
            canvas.height = window.innerHeight * dpr;
            canvas.style.width = `${window.innerWidth}px`;
            canvas.style.height = `${window.innerHeight}px`;

            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.scale(dpr, dpr);
            }

            initParticles(window.innerWidth, window.innerHeight);
        };

        const handleMouseMove = (e: MouseEvent) => {
            mouseRef.current = { x: e.clientX, y: e.clientY };
        };

        handleResize();
        window.addEventListener('resize', handleResize);
        window.addEventListener('mousemove', handleMouseMove);

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            window.removeEventListener('resize', handleResize);
            window.removeEventListener('mousemove', handleMouseMove);
            cancelAnimationFrame(animationRef.current);
        };
    }, [animate, initParticles]);

    return (
        <canvas
            ref={canvasRef}
            className="neural-background"
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: 0,
                pointerEvents: 'none',
                background: 'linear-gradient(135deg, #0a0515 0%, #1a0a2e 30%, #16082a 60%, #0f0518 100%)'
            }}
        />
    );
}

export default NeuralBackground;
