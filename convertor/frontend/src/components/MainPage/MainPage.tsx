/**
 * MainPage Component
 * 
 * Neural Frontiers Lab landing page
 * Features:
 * - Bento Grid Layout for Research Cards (Featured Item)
 * - Premium Deep Styling
 * - AnimatePresence for smooth transitions
 */

import React, { Suspense, lazy } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SideNav } from './SideNav';
import { TopNav } from './TopNav';
import { Footer } from './Footer';
import { NeuralBackground } from './NeuralBackground';
import {
    ResearchCard,
    GeminiIcon,
    VisualIcon,
    BCIIcon,
    TeamIcon,
    NeuralIcon,
    StarIcon
} from './ResearchCard';

// Lazy load 3D component
const Brain3D = lazy(() => import('./Brain3D'));

// Loading placeholder
function BrainLoading(): React.ReactElement {
    return (
        <div className="brain-loading">
            <div className="brain-loading-spinner" />
            <span>Loading visualization...</span>
        </div>
    );
}

// Research cards data
// Using 5 items max as per user request ("maximum 5 less than blocks")
// First item is the "Big" one
const researchCards = [
    {
        id: 'gemini',
        category: 'LATEST ADVANCEMENTS',
        title: 'Gemini 3 - Our Most Intelligent Model',
        icon: <GeminiIcon />,
        variant: 'featured' as const,
        description: 'Next-generation multimodal reasoning capabilities.'
    },
    {
        id: 'vision',
        category: 'RESEARCH FOCUS',
        title: 'Visual Processing',
        icon: <VisualIcon />,
        variant: 'default' as const
    },
    {
        id: 'bci',
        category: 'BLOG',
        title: 'AI & BCI',
        icon: <BCIIcon />,
        variant: 'default' as const
    },
    {
        id: 'team',
        category: 'OUR TEAM',
        title: 'Meet the Minds',
        icon: <TeamIcon />,
        variant: 'highlight' as const
    },
    {
        id: 'premium',
        category: 'PREMIUM',
        title: 'Future Models',
        icon: <StarIcon />,
        variant: 'default' as const
    }
];

export function MainPage(): React.ReactElement {
    // Default to collapsed (true) as per user request "default has to come minimizer"
    const [isSidebarCollapsed, setIsSidebarCollapsed] = React.useState(true);

    const toggleSidebar = () => {
        setIsSidebarCollapsed(!isSidebarCollapsed);
    };

    return (
        <AnimatePresence mode="wait">
            <motion.div
                className="mainpage"
                data-collapsed={isSidebarCollapsed}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
            >
                {/* Neural network background */}
                <NeuralBackground particleCount={80} connectionDistance={150} speed={0.2} />

                {/* Left sidebar navigation */}
                <SideNav isCollapsed={isSidebarCollapsed} toggleSidebar={toggleSidebar} />

                {/* Main content area */}
                <div className="mainpage-content">
                    {/* Top navigation */}
                    <TopNav />

                    {/* Hero section - Fluid Layout */}
                    <section className="hero">
                        <div className="hero-text">
                            <motion.h1
                                className="hero-title"
                                initial={{ opacity: 0, y: 30 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.8, ease: "easeOut" }}
                            >
                                <span className="hero-title-line">PIONEERING RESEARCH</span>
                                <span className="hero-title-line">ON THE PATH TO <span className="gradient-text">AGI</span></span>
                            </motion.h1>
                            <motion.p
                                className="hero-subtitle"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: 0.5, duration: 0.8 }}
                            >
                                Our mission is to ensure AGI benefits all of humanity.
                            </motion.p>
                        </div>

                        {/* 3D Brain visualization */}
                        <motion.div
                            className="hero-brain"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 1, delay: 0.2 }}
                        >
                            <Suspense fallback={<BrainLoading />}>
                                <Brain3D />
                            </Suspense>
                        </motion.div>
                    </section>

                    {/* Bento Grid layout for cards */}
                    <section className="cards-section">
                        <div className="bento-grid">
                            {researchCards.map((card, index) => (
                                <motion.div
                                    key={card.id}
                                    className={`bento-item ${index === 0 ? 'bento-item-large' : ''}`}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.6 + (index * 0.1), duration: 0.5 }}
                                >
                                    <ResearchCard
                                        category={card.category}
                                        title={card.title}
                                        icon={card.icon}
                                        variant={card.variant}
                                        className="h-full" // Ensure card fills bento cell
                                    />
                                </motion.div>
                            ))}
                        </div>
                    </section>

                    {/* Premium Footer */}
                    <Footer />
                </div>
            </motion.div>
        </AnimatePresence>
    );
}

export default MainPage;
