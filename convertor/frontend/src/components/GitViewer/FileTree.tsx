/**
 * FileTree Component
 * ===================
 * Virtualized file tree navigation for large repositories.
 * Recursive rendering with expand/collapse animations.
 */

import { useState, useCallback, useMemo, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { TreeNode } from './types';

interface FileTreeProps {
    nodes: TreeNode[];
    selectedPath: string | null;
    onSelect: (path: string) => void;
}

interface TreeNodeRowProps {
    node: TreeNode;
    depth: number;
    selectedPath: string | null;
    onSelect: (path: string) => void;
}

/** Get file icon based on extension */
function getFileIcon(name: string): { icon: string; className: string } {
    const ext = name.split('.').pop()?.toLowerCase() || '';

    const icons: Record<string, { icon: string; className: string }> = {
        md: { icon: 'M', className: 'tree-node__icon--md' },
        py: { icon: 'Py', className: 'tree-node__icon--py' },
        ts: { icon: 'TS', className: 'tree-node__icon--ts' },
        tsx: { icon: 'TS', className: 'tree-node__icon--ts' },
        js: { icon: 'JS', className: 'tree-node__icon--js' },
        jsx: { icon: 'JS', className: 'tree-node__icon--js' },
        json: { icon: '{ }', className: 'tree-node__icon--file' },
        yaml: { icon: 'Y', className: 'tree-node__icon--file' },
        yml: { icon: 'Y', className: 'tree-node__icon--file' },
    };

    return icons[ext] || { icon: '📄', className: 'tree-node__icon--file' };
}

/** Memoized tree node row */
const TreeNodeRow = memo(function TreeNodeRow({
    node,
    depth,
    selectedPath,
    onSelect
}: TreeNodeRowProps) {
    const [isExpanded, setIsExpanded] = useState(depth < 2);
    const isSelected = selectedPath === node.path;
    const isDirectory = node.type === 'directory';
    const hasChildren = isDirectory && node.children && node.children.length > 0;

    const handleClick = useCallback(() => {
        if (isDirectory) {
            setIsExpanded(prev => !prev);
        } else {
            onSelect(node.path);
        }
    }, [isDirectory, node.path, onSelect]);

    const fileIcon = useMemo(() =>
        isDirectory ? null : getFileIcon(node.name),
        [isDirectory, node.name]
    );

    return (
        <div className="tree-node" style={{ paddingLeft: depth * 12 }}>
            <div
                className={`tree-node__row ${isSelected ? 'tree-node__row--selected' : ''}`}
                onClick={handleClick}
            >
                {/* Chevron */}
                <svg
                    className={`tree-node__chevron ${isExpanded ? 'tree-node__chevron--expanded' : ''} ${!hasChildren ? 'tree-node__chevron--hidden' : ''}`}
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                >
                    <polyline points="9 18 15 12 9 6" />
                </svg>

                {/* Icon */}
                {isDirectory ? (
                    <svg className="tree-node__icon tree-node__icon--folder" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M10 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-8l-2-2z" />
                    </svg>
                ) : (
                    <span className={`tree-node__icon ${fileIcon?.className}`}>
                        {fileIcon?.icon}
                    </span>
                )}

                {/* Name */}
                <span className="tree-node__name" title={node.path}>
                    {node.name}
                </span>
            </div>

            {/* Children */}
            <AnimatePresence>
                {hasChildren && isExpanded && (
                    <motion.div
                        className="tree-node__children"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.2 }}
                    >
                        {node.children!.map(child => (
                            <TreeNodeRow
                                key={child.path}
                                node={child}
                                depth={depth + 1}
                                selectedPath={selectedPath}
                                onSelect={onSelect}
                            />
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
});

export function FileTree({ nodes, selectedPath, onSelect }: FileTreeProps) {
    // Sort: directories first, then alphabetically
    const sortedNodes = useMemo(() => {
        return [...nodes].sort((a, b) => {
            if (a.type !== b.type) {
                return a.type === 'directory' ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
        });
    }, [nodes]);

    return (
        <motion.div
            className="git-card file-tree"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
        >
            <div className="git-card__header">
                <svg className="git-card__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                </svg>
                <h2 className="git-card__title">Files</h2>
            </div>

            <div className="file-tree__content">
                {sortedNodes.length === 0 ? (
                    <div className="doc-preview__empty">
                        <p className="doc-preview__empty-subtitle">No files indexed yet</p>
                    </div>
                ) : (
                    sortedNodes.map(node => (
                        <TreeNodeRow
                            key={node.path}
                            node={node}
                            depth={0}
                            selectedPath={selectedPath}
                            onSelect={onSelect}
                        />
                    ))
                )}
            </div>
        </motion.div>
    );
}
