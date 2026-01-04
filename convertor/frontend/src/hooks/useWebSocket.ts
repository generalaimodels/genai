/**
 * WebSocket Hook for Real-Time File Updates
 * 
 * Features:
 * - Automatic connection to backend WebSocket endpoint
 * - Exponential backoff reconnection strategy
 * - Connection state management
 * - Message type discrimination
 * - Automatic cleanup on unmount
 * 
 * Usage:
 * ```tsx
 * const { isConnected, lastMessage } = useWebSocket();
 * 
 * useEffect(() => {
 *   if (lastMessage?.type === 'file_changed') {
 *     handleFileUpdate(lastMessage);
 *   }
 * }, [lastMessage]);
 * ```
 */

import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * WebSocket message types from backend
 */
export interface FileChangeMessage {
    type: 'file_changed' | 'connected' | 'heartbeat';
    path?: string;  // Present for file_changed
    action?: 'modified' | 'created' | 'deleted';  // Present for file_changed
    timestamp?: number;
    connection_id?: string;  // Present for connected
}

/**
 * Custom hook for WebSocket connection with auto-reconnect
 * 
 * @returns Object with connection state and last received message
 */
export function useWebSocket() {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<FileChangeMessage | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | null>(null);
    const reconnectAttempts = useRef(0);
    const mountedRef = useRef(true);

    const connect = useCallback(() => {
        // Don't connect if unmounted
        if (!mountedRef.current) return;

        // WebSocket URL (use ws:// for localhost, wss:// for production)
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.hostname}:8000/api/ws`;

        try {
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('[WebSocket] Connected to server');
                setIsConnected(true);
                reconnectAttempts.current = 0;
            };

            ws.onmessage = (event) => {
                try {
                    const message: FileChangeMessage = JSON.parse(event.data);
                    console.log('[WebSocket] Received message:', message);
                    setLastMessage(message);
                } catch (error) {
                    console.error('[WebSocket] Error parsing message:', error);
                }
            };

            ws.onclose = () => {
                console.log('[WebSocket] Connection closed');
                setIsConnected(false);

                // Only reconnect if still mounted
                if (mountedRef.current) {
                    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
                    const delay = Math.min(
                        1000 * Math.pow(2, reconnectAttempts.current),
                        30000
                    );

                    console.log(`[WebSocket] Reconnecting in ${delay / 1000}s (attempt ${reconnectAttempts.current + 1})`);

                    reconnectAttempts.current++;
                    reconnectTimeoutRef.current = window.setTimeout(connect, delay);
                }
            };

            ws.onerror = (error) => {
                console.error('[WebSocket] Error:', error);
                // onclose will be called after onerror, which will trigger reconnect
            };

            wsRef.current = ws;

        } catch (error) {
            console.error('[WebSocket] Connection error:', error);

            // Retry connection
            if (mountedRef.current) {
                const delay = Math.min(
                    1000 * Math.pow(2, reconnectAttempts.current),
                    30000
                );
                reconnectAttempts.current++;
                reconnectTimeoutRef.current = window.setTimeout(connect, delay);
            }
        }
    }, []);

    useEffect(() => {
        // Mark as mounted
        mountedRef.current = true;

        // Initial connection
        connect();

        // Cleanup on unmount
        return () => {
            mountedRef.current = false;

            // Close WebSocket connection
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }

            // Clear reconnection timeout
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
                reconnectTimeoutRef.current = null;
            }
        };
    }, [connect]);

    return {
        isConnected,
        lastMessage,
    };
}

export default useWebSocket;
