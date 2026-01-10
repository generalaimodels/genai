/**
 * PerformanceStats Component
 * 
 * Displays real-time backend performance metrics.
 * Showcases Phase 1 SOTA optimizations.
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface StatsData {
    total_docs: number;
    cached_conversions: number;
    total_accesses: number;
    avg_conversion_ms: number;
    hash_lookups?: number;
    bloom_hit_rate?: number;
}

export function PerformanceStats(): React.ReactElement {
    const [stats, setStats] = useState<StatsData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                setStats(data.stats || null);
                setLoading(false);
            } catch (error) {
                console.error('Failed to fetch stats:', error);
                setLoading(false);
            }
        };

        fetchStats();
        const interval = setInterval(fetchStats, 5000); // Update every 5s

        return () => clearInterval(interval);
    }, []);

    if (loading || !stats) {
        return <></>;
    }

    return (
        <motion.div
            className="performance-stats"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
        >
            <h3 className="stats-title">SOTA Performance</h3>
            <div className="stats-grid">
                <StatCard
                    label="Documents"
                    value={stats.total_docs.toLocaleString()}
                    icon="📄"
                    subtitle="Indexed files"
                />
                <StatCard
                    label="Cached"
                    value={stats.cached_conversions?.toLocaleString() || '0'}
                    icon="⚡"
                    subtitle="Conversions cached"
                />
                <StatCard
                    label="Speed"
                    value={`${stats.avg_conversion_ms?.toFixed(1) || '0'}ms`}
                    icon="🚀"
                    subtitle="Avg conversion time"
                />
                {stats.bloom_hit_rate !== undefined && (
                    <StatCard
                        label="Cache Hit"
                        value={`${(stats.bloom_hit_rate * 100).toFixed(1)}%`}
                        icon="🎯"
                        subtitle="Bloom filter efficiency"
                    />
                )}
            </div>
        </motion.div>
    );
}

interface StatCardProps {
    label: string;
    value: string;
    icon: string;
    subtitle: string;
}

function StatCard({ label, value, icon, subtitle }: StatCardProps): React.ReactElement {
    return (
        <motion.div
            className="stat-card"
            whileHover={{ scale: 1.03, y: -2 }}
            transition={{ duration: 0.2 }}
        >
            <div className="stat-icon">{icon}</div>
            <div className="stat-content">
                <div className="stat-value">{value}</div>
                <div className="stat-label">{label}</div>
                <div className="stat-subtitle">{subtitle}</div>
            </div>
        </motion.div>
    );
}

export default PerformanceStats;
