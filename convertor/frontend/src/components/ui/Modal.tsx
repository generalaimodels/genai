/**
 * Modal Component
 * 
 * Accessible modal with premium animations.
 * 
 * Features:
 * - Framer Motion enter/exit animations
 * - Backdrop blur
 * - Keyboard handling (ESC to close)
 * - Focus trap
 * - Portal rendering
 */

import React, { useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import type { ModalProps } from '@/types';

const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
};

const modalVariants = {
    hidden: {
        opacity: 0,
        scale: 0.95,
        y: -20,
    },
    visible: {
        opacity: 1,
        scale: 1,
        y: 0,
        transition: {
            type: 'spring',
            damping: 25,
            stiffness: 300,
        },
    },
    exit: {
        opacity: 0,
        scale: 0.95,
        y: -20,
        transition: {
            duration: 0.15,
        },
    },
};

const sizeClasses: Record<string, string> = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
};

export function Modal({
    isOpen,
    onClose,
    children,
    title,
    size = 'md',
}: ModalProps): React.ReactElement | null {
    const modalRef = useRef<HTMLDivElement>(null);

    // Handle ESC key
    const handleKeyDown = useCallback(
        (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        },
        [onClose]
    );

    // Handle backdrop click
    const handleBackdropClick = useCallback(
        (event: React.MouseEvent) => {
            if (event.target === event.currentTarget) {
                onClose();
            }
        },
        [onClose]
    );

    // Add/remove event listeners
    useEffect(() => {
        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown);
            document.body.style.overflow = 'hidden';
        }

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
        };
    }, [isOpen, handleKeyDown]);

    // Focus trap
    useEffect(() => {
        if (isOpen && modalRef.current) {
            const focusableElements = modalRef.current.querySelectorAll<HTMLElement>(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            const firstElement = focusableElements[0];
            firstElement?.focus();
        }
    }, [isOpen]);

    const modalContent = (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    className="modal-backdrop"
                    variants={backdropVariants}
                    initial="hidden"
                    animate="visible"
                    exit="hidden"
                    onClick={handleBackdropClick}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        zIndex: 1000,
                        display: 'flex',
                        alignItems: 'flex-start',
                        justifyContent: 'center',
                        paddingTop: '15vh',
                        background: 'var(--surface-overlay)',
                        backdropFilter: 'blur(4px)',
                    }}
                >
                    <motion.div
                        ref={modalRef}
                        className={`modal-content ${sizeClasses[size]}`}
                        variants={modalVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby={title ? 'modal-title' : undefined}
                        style={{
                            width: '100%',
                            background: 'var(--bg-elevated)',
                            borderRadius: 'var(--radius-2xl)',
                            boxShadow: 'var(--shadow-2xl)',
                            overflow: 'hidden',
                            border: '1px solid var(--border-primary)',
                        }}
                    >
                        {title && (
                            <div className="modal-header" style={{ padding: 'var(--space-4) var(--space-5)', borderBottom: '1px solid var(--border-primary)' }}>
                                <h2 id="modal-title" style={{ fontSize: 'var(--text-lg)', fontWeight: 600 }}>
                                    {title}
                                </h2>
                            </div>
                        )}
                        {children}
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );

    // Render to portal
    if (typeof document === 'undefined') return null;

    const portalRoot = document.getElementById('modal-root') || document.body;
    return createPortal(modalContent, portalRoot);
}

export default Modal;
