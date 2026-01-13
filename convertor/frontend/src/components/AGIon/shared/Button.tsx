/**
 * Button Component - AGIon Design System
 * 
 * VARIANTS:
 * - primary: Solid accent gradient, white text
 * - secondary: Border outline, accent color
 * - ghost: No border, text only
 * 
 * INTERACTION:
 * - Lift on hover (translateY -1px)
 * - Accent shadow on hover
 * - Active state (no lift)
 * - Disabled state (reduced opacity)
 */

import React from 'react';
import './Button.css';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  children: React.ReactNode;
  onClick?: () => void;
  href?: string;
  type?: 'button' | 'submit' | 'reset';
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  children,
  onClick,
  href,
  type = 'button',
  disabled = false,
  className = '',
  ariaLabel,
}) => {
  const baseClass = 'agion-button';
  const variantClass = `agion-button--${variant}`;
  const sizeClass = `agion-button--${size}`;
  const disabledClass = disabled ? 'agion-button--disabled' : '';
  
  const classNames = [baseClass, variantClass, sizeClass, disabledClass, className]
    .filter(Boolean)
    .join(' ');

  // If href is provided, render as anchor
  if (href) {
    return (
      <a
        href={href}
        className={classNames}
        aria-label={ariaLabel}
        onClick={(e) => {
          if (disabled) {
            e.preventDefault();
            return;
          }
          onClick?.();
        }}
      >
        {children}
      </a>
    );
  }

  // Render as button
  return (
    <button
      type={type}
      className={classNames}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
};
