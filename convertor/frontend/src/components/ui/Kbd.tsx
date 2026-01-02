/**
 * Kbd Component
 * 
 * Keyboard key indicator with premium styling.
 */

import React from 'react';

interface KbdProps {
    children: React.ReactNode;
    className?: string;
}

export function Kbd({ children, className = '' }: KbdProps): React.ReactElement {
    return (
        <kbd
            className={`keyboard-key ${className}`}
            style={{
                display: 'inline-block',
                padding: 'var(--space-1) var(--space-2)',
                fontFamily: 'var(--font-sans)',
                fontSize: 'var(--text-xs)',
                fontWeight: 500,
                lineHeight: 1,
                color: 'var(--text-secondary)',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-primary)',
                borderBottomWidth: '2px',
                borderRadius: 'var(--radius-sm)',
                boxShadow: 'inset 0 -1px 0 var(--border-secondary)',
            }}
        >
            {children}
        </kbd>
    );
}

export default Kbd;
