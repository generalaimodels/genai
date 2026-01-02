/**
 * Button Component
 * 
 * Premium button with multiple variants, sizes, and loading states.
 * Features smooth Framer Motion animations.
 */

import React from 'react';
import { motion, type HTMLMotionProps } from 'framer-motion';

interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'ref' | 'children'> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg';
    isLoading?: boolean;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
    children?: React.ReactNode;
}

const variantStyles: Record<string, string> = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    ghost: 'btn-ghost',
    danger: 'btn-danger',
};

const sizeStyles: Record<string, string> = {
    sm: 'btn-sm',
    md: 'btn-md',
    lg: 'btn-lg',
};

export function Button({
    variant = 'primary',
    size = 'md',
    isLoading = false,
    leftIcon,
    rightIcon,
    children,
    className = '',
    disabled,
    ...props
}: ButtonProps): React.ReactElement {
    const isDisabled = disabled || isLoading;

    return (
        <motion.button
            className={`btn ${variantStyles[variant] ?? ''} ${sizeStyles[size] ?? ''} ${className}`}
            disabled={isDisabled}
            whileHover={{ scale: isDisabled ? 1 : 1.02 }}
            whileTap={{ scale: isDisabled ? 1 : 0.98 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            {...props}
        >
            {isLoading ? (
                <span className="btn-spinner" aria-hidden="true">
                    <svg
                        className="animate-spin"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                    >
                        <circle
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="3"
                            strokeLinecap="round"
                            strokeDasharray="60"
                            strokeDashoffset="20"
                        />
                    </svg>
                </span>
            ) : (
                leftIcon && <span className="btn-icon-left">{leftIcon}</span>
            )}
            <span className="btn-text">{children}</span>
            {!isLoading && rightIcon && <span className="btn-icon-right">{rightIcon}</span>}
        </motion.button>
    );
}

export default Button;
