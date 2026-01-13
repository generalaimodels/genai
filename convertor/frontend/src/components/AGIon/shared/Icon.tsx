/**
 * Icon Component - AGIon Design System
 * 
 * FEATURES:
 * - Inline SVG for performance
 * - 16px stroke icons (Feather/Lucide style)
 * - Accessible with aria-label
 * - Customizable size and color
 */

import React from 'react';

export type IconName =
    | 'research'
    | 'system'
    | 'community'
    | 'blueprint'
    | 'book'
    | 'check'
    | 'shield'
    | 'arrow-right'
    | 'github'
    | 'x'
    | 'linkedin'
    | 'menu'
    | 'close'
    | 'fragment'
    | 'alert'
    | 'disconnect';

export interface IconProps {
    name: IconName;
    size?: number;
    color?: string;
    className?: string;
    ariaLabel?: string;
}

export const Icon: React.FC<IconProps> = ({
    name,
    size = 16,
    color = 'currentColor',
    className = '',
    ariaLabel,
}) => {
    const iconPaths: Record<IconName, JSX.Element> = {
        research: (
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" strokeLinecap="round" strokeLinejoin="round" />
        ),
        system: (
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" strokeLinecap="round" strokeLinejoin="round" />
        ),
        community: (
            <g>
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="9" cy="7" r="4" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" strokeLinecap="round" strokeLinejoin="round" />
            </g>
        ),
        blueprint: (
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8" strokeLinecap="round" strokeLinejoin="round" />
        ),
        book: (
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20 M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" strokeLinecap="round" strokeLinejoin="round" />
        ),
        check: (
            <polyline points="20 6 9 17 4 12" strokeLinecap="round" strokeLinejoin="round" />
        ),
        shield: (
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" strokeLinecap="round" strokeLinejoin="round" />
        ),
        'arrow-right': (
            <path d="M5 12h14 M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
        ),
        github: (
            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" strokeLinecap="round" strokeLinejoin="round" />
        ),
        x: (
            <path d="M4 4l16 16m0-16L4 20" strokeLinecap="round" strokeLinejoin="round" />
        ),
        linkedin: (
            <g>
                <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6zM2 9h4v12H2z" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="4" cy="4" r="2" strokeLinecap="round" strokeLinejoin="round" />
            </g>
        ),
        menu: (
            <path d="M3 12h18M3 6h18M3 18h18" strokeLinecap="round" strokeLinejoin="round" />
        ),
        close: (
            <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" />
        ),
        fragment: (
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" strokeLinecap="round" strokeLinejoin="round" />
        ),
        alert: (
            <g>
                <circle cx="12" cy="12" r="10" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M12 8v4M12 16h.01" strokeLinecap="round" strokeLinejoin="round" />
            </g>
        ),
        disconnect: (
            <path d="M17 11 7 11 M2 8l4-4 4 4 M22 16l-4 4-4-4 M22 8l-4-4-4 4 M2 16l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
        ),
    };

    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            stroke={color}
            strokeWidth="2"
            className={`agion-icon ${className}`}
            aria-label={ariaLabel}
            role={ariaLabel ? 'img' : 'presentation'}
        >
            {iconPaths[name]}
        </svg>
    );
};
