/**
 * Card Component - AGIon Design System
 * 
 * FEATURES:
 * - Glass effect variant
 * - Hover lift animation (4px translateY)
 * - Border color change on hover
 * - Flexible padding and corner radius
 */

import React from 'react';
import './Card.css';

export interface CardProps {
    children: React.ReactNode;
    variant?: 'default' | 'glass';
    hoverable?: boolean;
    padding?: 'sm' | 'md' | 'lg';
    className?: string;
    onClick?: () => void;
}

export const Card: React.FC<CardProps> = ({
    children,
    variant = 'default',
    hoverable = false,
    padding = 'md',
    className = '',
    onClick,
}) => {
    const baseClass = 'agion-card';
    const variantClass = `agion-card--${variant}`;
    const hoverClass = hoverable ? 'agion-card--hoverable' : '';
    const paddingClass = `agion-card--padding-${padding}`;
    const clickableClass = onClick ? 'agion-card--clickable' : '';

    const classNames = [baseClass, variantClass, hoverClass, paddingClass, clickableClass, className]
        .filter(Boolean)
        .join(' ');

    return (
        <div className={classNames} onClick={onClick}>
            {children}
        </div>
    );
};
