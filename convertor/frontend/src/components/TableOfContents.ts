/**
 * Table of Contents Component
 * 
 * Generates TOC from document headings with:
 * - Scroll spy for active section
 * - Smooth scroll on click
 * - Nested levels
 */

import { Heading } from '../api/client';

let observer: IntersectionObserver | null = null;

/**
 * Render table of contents from headings
 */
export function renderToc(headings: Heading[]): void {
    const container = document.getElementById('toc-nav');
    const tocSection = document.getElementById('toc');

    if (!container || !tocSection) return;

    // Hide TOC if no headings
    if (headings.length === 0) {
        tocSection.style.display = 'none';
        return;
    }

    tocSection.style.display = '';
    container.innerHTML = '';

    // Create links for each heading
    for (const heading of headings) {
        const link = document.createElement('a');
        link.className = 'toc-link';
        link.href = `#${heading.id}`;
        link.textContent = heading.text;
        link.dataset.level = String(heading.level);
        link.dataset.id = heading.id;

        link.addEventListener('click', (e) => {
            e.preventDefault();
            scrollToHeading(heading.id);
        });

        container.appendChild(link);
    }

    // Set up scroll spy
    setupScrollSpy(headings);
}

/**
 * Smooth scroll to heading
 */
function scrollToHeading(id: string): void {
    const element = document.getElementById(id);
    if (!element) return;

    // Update URL hash without scrolling
    history.pushState(null, '', `#${id}`);

    // Smooth scroll to element
    element.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Highlight the link
    setActiveTocItem(id);
}

/**
 * Set up intersection observer for scroll spy
 */
function setupScrollSpy(headings: Heading[]): void {
    // Cleanup previous observer
    if (observer) {
        observer.disconnect();
    }

    const headingElements = headings
        .map((h) => document.getElementById(h.id))
        .filter((el): el is HTMLElement => el !== null);

    if (headingElements.length === 0) return;

    // Track which heading is visible
    const visibleHeadings = new Set<string>();

    observer = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting) {
                    visibleHeadings.add(entry.target.id);
                } else {
                    visibleHeadings.delete(entry.target.id);
                }
            }

            // Find the first visible heading (top-most)
            for (const heading of headings) {
                if (visibleHeadings.has(heading.id)) {
                    setActiveTocItem(heading.id);
                    break;
                }
            }
        },
        {
            rootMargin: '-80px 0px -80% 0px',
            threshold: 0,
        }
    );

    for (const element of headingElements) {
        observer.observe(element);
    }
}

/**
 * Set active TOC item
 */
function setActiveTocItem(id: string): void {
    // Remove active from all
    document.querySelectorAll('.toc-link.active').forEach((el) => {
        el.classList.remove('active');
    });

    // Add active to matching
    const activeLink = document.querySelector(`.toc-link[data-id="${id}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }
}

/**
 * Clear TOC
 */
export function clearToc(): void {
    const container = document.getElementById('toc-nav');
    if (container) {
        container.innerHTML = '';
    }

    if (observer) {
        observer.disconnect();
        observer = null;
    }
}
