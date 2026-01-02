/**
 * ThemeToggle Component
 * 
 * Animated sun/moon toggle for theme switching.
 * Features premium rotation animation on toggle.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { useTheme } from '@/hooks';

export function ThemeToggle(): React.ReactElement {
    const { toggleTheme, isDark } = useTheme();

    return (
        <motion.button
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            whileHover={{ scale: 1.05, rotate: 10 }}
            whileTap={{ scale: 0.95 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
        >
            {/* Sun Icon */}
            <motion.svg
                className="sun-icon"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                initial={false}
                animate={{
                    scale: isDark ? 1 : 0,
                    rotate: isDark ? 0 : -90,
                    opacity: isDark ? 1 : 0,
                }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                style={{ position: 'absolute' }}
            >
                <circle cx="12" cy="12" r="5" />
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
            </motion.svg>

            {/* Moon Icon */}
            <motion.svg
                className="moon-icon"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                initial={false}
                animate={{
                    scale: isDark ? 0 : 1,
                    rotate: isDark ? 90 : 0,
                    opacity: isDark ? 0 : 1,
                }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                style={{ position: 'absolute' }}
            >
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </motion.svg>
        </motion.button>
    );
}

export default ThemeToggle;
