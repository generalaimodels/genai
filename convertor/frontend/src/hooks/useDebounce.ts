/**
 * useDebounce Hook
 * 
 * Debounces a value for optimized performance.
 * Perfect for search inputs and API calls.
 * 
 * Features:
 * - Configurable delay
 * - Cleanup on unmount
 * - Generic type support
 */

import { useState, useEffect, useRef, useCallback } from 'react';

export function useDebounce<T>(value: T, delay: number = 300): T {
    const [debouncedValue, setDebouncedValue] = useState<T>(value);

    useEffect(() => {
        // Set up debounce timer
        const timer = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        // Clean up timer on value change or unmount
        return () => {
            clearTimeout(timer);
        };
    }, [value, delay]);

    return debouncedValue;
}

/**
 * useDebouncedCallback Hook
 * 
 * Returns a debounced version of the callback function.
 * Useful for event handlers that trigger expensive operations.
 */
export function useDebouncedCallback<T extends (...args: unknown[]) => void>(
    callback: T,
    delay: number = 300
): { debouncedFn: (...args: Parameters<T>) => void; cancel: () => void } {
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const callbackRef = useRef(callback);

    // Keep callback ref updated
    useEffect(() => {
        callbackRef.current = callback;
    }, [callback]);

    const cancel = useCallback(() => {
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }
    }, []);

    const debouncedFn = useCallback(
        (...args: Parameters<T>) => {
            cancel();

            timeoutRef.current = setTimeout(() => {
                callbackRef.current(...args);
            }, delay);
        },
        [delay, cancel]
    );

    // Cleanup on unmount
    useEffect(() => {
        return cancel;
    }, [cancel]);

    return { debouncedFn, cancel };
}

export default useDebounce;
