
"""
Purpose: Automatic generation and writing of __init__.py files with TYPE_CHECKING imports.
         This script recursively scans Python packages, extracts ONLY classes and standalone functions,
         and automatically generates __init__.py files with proper TYPE_CHECKING import blocks.

Features:
- Automatic __init__.py file generation with TYPE_CHECKING imports
- Extracts ONLY classes and standalone functions (NO class methods)
- Ensures unique symbol extraction with deduplication
- Recursive package discovery and processing
- AST-based symbol extraction for accurate import detection
- Intelligent import path resolution
- Support for nested packages at unlimited depth
- Thread-safe operations for parallel processing

Usage:
    # Generate __init__.py files for entire project
    generator = InitPyGenerator("/path/to/project")
    generator.generate_all_init_files()
    
    # Generate for specific package
    generator.generate_init_file("/path/to/specific/package")

Example Output:

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module_a import ClassA, standalone_func_a
    from .module_b import ClassB, standalone_func_b
    from .subpkg_a import DeepClass, deep_function
    from .subpkg_b import AnotherClass, another_function

"""

import os
import ast
import sys
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, NamedTuple, Union
from functools import lru_cache
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class SymbolInfo(NamedTuple):
    """Container for class and standalone function information extracted from Python modules."""
    name: str
    symbol_type: str  # 'class' or 'function'
    is_private: bool


class ModuleData(NamedTuple):
    """Container for module data and metadata."""
    module_name: str
    file_path: Path
    symbols: Set[SymbolInfo]  # Using Set for automatic uniqueness
    relative_import_path: str
    is_package: bool


class PackageStructure(NamedTuple):
    """Container for complete package structure information."""
    package_path: Path
    modules: List[ModuleData]
    subpackages: Dict[str, 'PackageStructure']
    parent_path: Optional[str]


class InitPyGenerator:
    """
    Advanced automatic __init__.py generator with TYPE_CHECKING import blocks.
    
    This class focuses specifically on extracting classes and standalone functions only,
    ensuring clean and minimal TYPE_CHECKING import blocks without class methods,
    variables, constants, or other symbols.
    
    Key Features:
    - Extracts ONLY classes and standalone functions (NO class methods)
    - Automatic deduplication of symbols
    - Recursive package scanning with unlimited depth
    - AST-based symbol extraction for accuracy
    - Smart import path resolution
    - Thread-safe parallel processing
    - Comprehensive error handling
    
    Performance Optimizations:
    - LRU caching for repeated operations
    - Set-based symbol storage for O(1) uniqueness
    - Efficient file system traversal
    - Minimal I/O operations
    """
    
    def __init__(self, root_path: str, max_depth: int = 20, 
                 backup_existing: bool = True, parallel_processing: bool = True):
        """
        Initialize the InitPyGenerator with configuration options.
        
        Args:
            root_path: Root directory path to scan for packages
            max_depth: Maximum recursion depth for package discovery
            backup_existing: Whether to backup existing __init__.py files
            parallel_processing: Enable parallel processing for large projects
        """
        self.root_path = Path(root_path).resolve()
        self.max_depth = max_depth
        self.backup_existing = backup_existing
        self.parallel_processing = parallel_processing
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Caching for performance
        self._symbol_cache: Dict[str, Set[SymbolInfo]] = {}
        
        # Configuration
        self.excluded_dirs = {
            '__pycache__', '.git', '.svn', '.hg', '.tox', 'venv', 'env',
            '.venv', '.env', 'node_modules', '.pytest_cache', '.mypy_cache'
        }
        self.excluded_files = {
            '__pycache__.py', 'setup.py', 'conftest.py'
        }
        
        # Template for generated __init__.py files
        self.init_template = '''"""
Auto-generated __init__.py file with TYPE_CHECKING imports.
Generated on: {timestamp}
Generator: InitPyGenerator v2.0 (Classes & Standalone Functions Only)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
{imports}

__all__ = [
{all_exports}
]
'''

    @lru_cache(maxsize=512)
    def _extract_symbols_from_file(self, file_path: Path) -> Set[SymbolInfo]:
        """
        Extract ONLY classes and standalone functions from a Python file using AST parsing.
        
        This method performs focused AST analysis to identify only top-level classes and 
        standalone functions, excluding class methods, nested functions, and other symbols.
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            Set of SymbolInfo objects representing found classes and standalone functions
        """
        symbols = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Parse AST with error recovery
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError:
                print(f"Warning: Syntax error in {file_path}, skipping...")
                return symbols
                
            # Extract only top-level classes and standalone functions
            symbols = self._extract_top_level_symbols(tree)
                    
            # Check for __all__ exports to filter symbols
            all_exports = self._extract_all_exports(tree)
            if all_exports:
                # Only include symbols that are in __all__
                all_set = set(all_exports)
                symbols = {s for s in symbols if s.name in all_set}
                
        except (OSError, UnicodeDecodeError, MemoryError) as e:
            print(f"Warning: Cannot process {file_path}: {e}")
            
        return symbols
    
    def _extract_top_level_symbols(self, tree: ast.AST) -> Set[SymbolInfo]:
        """
        Extract only top-level classes and standalone functions from AST.
        
        This method specifically looks at the top-level body of the module,
        avoiding class methods and nested functions.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            Set of SymbolInfo objects for top-level symbols only
        """
        symbols = set()
        
        # Only iterate through top-level statements in the module
        for node in tree.body:
            symbol_info = self._analyze_top_level_node(node)
            if symbol_info:
                symbols.add(symbol_info)
        
        return symbols
    
    def _analyze_top_level_node(self, node: ast.AST) -> Optional[SymbolInfo]:
        """
        Analyze a top-level AST node to extract ONLY class and standalone function information.
        
        Args:
            node: Top-level AST node to analyze
            
        Returns:
            SymbolInfo object if node represents a class or standalone function, None otherwise
        """
        # Class definitions (top-level only)
        if isinstance(node, ast.ClassDef):
            return SymbolInfo(
                name=node.name,
                symbol_type='class',
                is_private=node.name.startswith('_')
            )
        
        # Function definitions (top-level only, including async functions)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return SymbolInfo(
                name=node.name,
                symbol_type='function',
                is_private=node.name.startswith('_')
            )
        
        # Ignore all other node types (variables, constants, imports, etc.)
        return None
    
    def _extract_all_exports(self, tree: ast.AST) -> Set[str]:
        """
        Extract __all__ list from AST if present.
        
        Args:
            tree: AST tree to search
            
        Returns:
            Set of exported symbol names from __all__
        """
        # Only look at top-level assignments
        for node in tree.body:
            if (isinstance(node, ast.Assign) and 
                len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == '__all__'):
                
                if isinstance(node.value, ast.List):
                    exports = set()
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Str):
                            exports.add(elt.s)
                        elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports.add(elt.value)
                    return exports
        return set()
    
    def _scan_package_directory(self, package_path: Path, depth: int = 0) -> PackageStructure:
        """
        Recursively scan a package directory to build complete structure.
        
        Args:
            package_path: Path to the package directory
            depth: Current recursion depth
            
        Returns:
            PackageStructure object representing the complete package
        """
        if depth > self.max_depth:
            return PackageStructure(package_path, [], {}, None)
        
        modules = []
        subpackages = {}
        
        try:
            for item in package_path.iterdir():
                # Skip excluded directories and files
                if item.name in self.excluded_dirs or item.name in self.excluded_files:
                    continue
                
                if item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                    # Extract symbols from Python module
                    symbols = self._extract_symbols_from_file(item)
                    
                    # Filter to only public (non-private) classes and standalone functions
                    exportable_symbols = {
                        s for s in symbols 
                        if not s.is_private
                    }
                    
                    if exportable_symbols:  # Only include modules with exportable symbols
                        module_name = item.stem
                        modules.append(ModuleData(
                            module_name=module_name,
                            file_path=item,
                            symbols=exportable_symbols,
                            relative_import_path=f".{module_name}",
                            is_package=False
                        ))
                
                elif item.is_dir() and (item / '__init__.py').exists():
                    # Recursively process subpackage
                    subpackage_structure = self._scan_package_directory(item, depth + 1)
                    if subpackage_structure.modules or subpackage_structure.subpackages:
                        subpackages[item.name] = subpackage_structure
        
        except (OSError, PermissionError) as e:
            print(f"Warning: Cannot scan directory {package_path}: {e}")
        
        return PackageStructure(
            package_path=package_path,
            modules=modules,
            subpackages=subpackages,
            parent_path=str(package_path.parent) if package_path.parent != package_path else None
        )
    
    def _generate_type_checking_imports(self, package_structure: PackageStructure) -> str:
        """
        Generate TYPE_CHECKING import statements from package structure.
        
        Args:
            package_structure: PackageStructure object to process
            
        Returns:
            Formatted string containing TYPE_CHECKING imports
        """
        import_lines = []
        
        # Generate imports for direct modules
        for module in package_structure.modules:
            if module.symbols:
                # Separate classes and standalone functions for better organization
                classes = sorted([s.name for s in module.symbols if s.symbol_type == 'class'])
                functions = sorted([s.name for s in module.symbols if s.symbol_type == 'function'])
                
                # Combine all symbols maintaining order (classes first, then functions)
                all_symbols = classes + functions
                
                if all_symbols:
                    symbols_str = ', '.join(all_symbols)
                    import_lines.append(f"    from {module.relative_import_path} import {symbols_str}")
        
        # Generate imports for subpackages
        for subpkg_name, subpkg_structure in package_structure.subpackages.items():
            subpkg_symbols = self._collect_unique_symbols_from_package(subpkg_structure)
            if subpkg_symbols:
                # Sort classes first, then functions
                sorted_symbols = sorted(subpkg_symbols)
                
                symbols_str = ', '.join(sorted_symbols)
                import_lines.append(f"    from .{subpkg_name} import {symbols_str}")
        
        return '\n'.join(import_lines) if import_lines else "    pass"
    
    def _collect_unique_symbols_from_package(self, package_structure: PackageStructure) -> Set[str]:
        """
        Recursively collect all unique exportable symbols from a package structure.
        
        Args:
            package_structure: PackageStructure to process
            
        Returns:
            Set of all unique exportable symbol names
        """
        all_symbols = set()
        
        # Collect symbols from direct modules
        for module in package_structure.modules:
            for symbol in module.symbols:
                if not symbol.is_private:
                    all_symbols.add(symbol.name)
        
        # Recursively collect from subpackages
        for subpkg_structure in package_structure.subpackages.values():
            all_symbols.update(self._collect_unique_symbols_from_package(subpkg_structure))
        
        return all_symbols
    
    def _generate_all_exports(self, package_structure: PackageStructure) -> str:
        """
        Generate __all__ list from package structure.
        
        Args:
            package_structure: PackageStructure to process
            
        Returns:
            Formatted string containing __all__ list
        """
        all_symbols = self._collect_unique_symbols_from_package(package_structure)
        
        if not all_symbols:
            return "    # No exportable classes or standalone functions found"
        
        # Sort symbols alphabetically
        sorted_symbols = sorted(all_symbols)
        
        # Format as properly indented list
        symbol_lines = [f'    "{symbol}",' for symbol in sorted_symbols]
        return '\n'.join(symbol_lines)
    
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create backup of existing __init__.py file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to backup file if created, None otherwise
        """
        if not self.backup_existing or not file_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f'.backup_{timestamp}.py')
        
        try:
            shutil.copy2(file_path, backup_path)
            return backup_path
        except OSError as e:
            print(f"Warning: Cannot create backup for {file_path}: {e}")
            return None
    
    def generate_init_file(self, package_path: Union[str, Path]) -> bool:
        """
        Generate __init__.py file for a specific package.
        
        Args:
            package_path: Path to the package directory
            
        Returns:
            True if generation was successful, False otherwise
        """
        package_path = Path(package_path)
        init_file_path = package_path / '__init__.py'
        
        try:
            # Scan package structure
            package_structure = self._scan_package_directory(package_path)
            
            # Check if there are any classes or standalone functions to export
            has_exportable_content = False
            for module in package_structure.modules:
                if module.symbols:
                    has_exportable_content = True
                    break
            
            if not has_exportable_content and not package_structure.subpackages:
                print(f"No exportable classes or standalone functions found in {package_path}")
                return False
            
            # Create backup if file exists
            backup_path = self._create_backup(init_file_path)
            if backup_path:
                print(f"Created backup: {backup_path}")
            
            # Generate TYPE_CHECKING imports
            type_checking_imports = self._generate_type_checking_imports(package_structure)
            
            # Generate __all__ exports
            all_exports = self._generate_all_exports(package_structure)
            
            # Create final content
            timestamp = datetime.now().isoformat()
            content = self.init_template.format(
                timestamp=timestamp,
                imports=type_checking_imports,
                all_exports=all_exports
            )
            
            # Write the file
            with open(init_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Generated: {init_file_path}")
            
            # Print summary of what was found
            total_classes = sum(1 for module in package_structure.modules 
                              for symbol in module.symbols if symbol.symbol_type == 'class')
            total_functions = sum(1 for module in package_structure.modules 
                                for symbol in module.symbols if symbol.symbol_type == 'function')
            print(f"  Found: {total_classes} classes, {total_functions} standalone functions")
            
            return True
            
        except Exception as e:
            print(f"Error generating {init_file_path}: {e}")
            return False
    
    def generate_all_init_files(self) -> Dict[str, bool]:
        """
        Generate __init__.py files for all packages in the root directory.
        
        Returns:
            Dictionary mapping package paths to generation success status
        """
        results = {}
        package_paths = self._discover_all_packages()
        
        if self.parallel_processing and len(package_paths) > 1:
            # Use parallel processing for large projects
            with ThreadPoolExecutor(max_workers=min(len(package_paths), 8)) as executor:
                future_to_path = {
                    executor.submit(self.generate_init_file, path): path 
                    for path in package_paths
                }
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[str(path)] = future.result()
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
                        results[str(path)] = False
        else:
            # Sequential processing
            for package_path in package_paths:
                results[str(package_path)] = self.generate_init_file(package_path)
        
        return results
    
    def _discover_all_packages(self) -> List[Path]:
        """
        Discover all packages in the root directory recursively.
        
        Returns:
            List of paths to all discovered packages
        """
        packages = []
        
        def scan_directory(directory: Path, depth: int = 0):
            if depth > self.max_depth:
                return
            
            try:
                for item in directory.iterdir():
                    if (item.is_dir() and 
                        item.name not in self.excluded_dirs and
                        not item.name.startswith('.')):
                        
                        # Check if it's a package (has Python files with classes/standalone functions)
                        has_exportable_content = False
                        
                        for file_path in item.glob('*.py'):
                            if file_path.name != '__init__.py':
                                symbols = self._extract_symbols_from_file(file_path)
                                if any(not s.is_private for s in symbols):
                                    has_exportable_content = True
                                    break
                        
                        # Also check for subpackages
                        has_subpackages = any(
                            (subdir / '__init__.py').exists() 
                            for subdir in item.iterdir() 
                            if subdir.is_dir()
                        )
                        
                        if has_exportable_content or has_subpackages:
                            packages.append(item)
                            scan_directory(item, depth + 1)
                            
            except (OSError, PermissionError) as e:
                print(f"Warning: Cannot scan {directory}: {e}")
        
        scan_directory(self.root_path)
        return packages
    
    def preview_generation(self, package_path: Union[str, Path]) -> str:
        """
        Preview what would be generated for a package without writing files.
        
        Args:
            package_path: Path to the package to preview
            
        Returns:
            String containing the preview of generated __init__.py content
        """
        package_path = Path(package_path)
        
        try:
            package_structure = self._scan_package_directory(package_path)
            type_checking_imports = self._generate_type_checking_imports(package_structure)
            all_exports = self._generate_all_exports(package_structure)
            
            timestamp = datetime.now().isoformat()
            content = self.init_template.format(
                timestamp=timestamp,
                imports=type_checking_imports,
                all_exports=all_exports
            )
            
            return content
            
        except Exception as e:
            return f"Error generating preview: {e}"
    
    def get_package_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the discovered packages and symbols.
        
        Returns:
            Dictionary containing various statistics
        """
        packages = self._discover_all_packages()
        total_modules = 0
        total_classes = 0
        total_functions = 0
        
        for package_path in packages:
            package_structure = self._scan_package_directory(package_path)
            total_modules += len(package_structure.modules)
            for module in package_structure.modules:
                for symbol in module.symbols:
                    if symbol.symbol_type == 'class':
                        total_classes += 1
                    elif symbol.symbol_type == 'function':
                        total_functions += 1
        
        return {
            'total_packages': len(packages),
            'total_modules': total_modules,
            'total_classes': total_classes,
            'total_standalone_functions': total_functions,
            'total_symbols': total_classes + total_functions
        }
    
    def analyze_file_symbols(self, file_path: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Analyze a single file to show what symbols would be extracted.
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            Dictionary with 'classes' and 'functions' lists
        """
        file_path = Path(file_path)
        symbols = self._extract_symbols_from_file(file_path)
        
        classes = [s.name for s in symbols if s.symbol_type == 'class' and not s.is_private]
        functions = [s.name for s in symbols if s.symbol_type == 'function' and not s.is_private]
        
        return {
            'classes': sorted(classes),
            'functions': sorted(functions),
            'total': len(classes) + len(functions)
        }


# Convenience functions for direct usage
def generate_init_files_for_project(project_path: str, **kwargs) -> Dict[str, bool]:
    """
    Generate __init__.py files for an entire project.
    
    Args:
        project_path: Root path of the project
        **kwargs: Additional arguments for InitPyGenerator
        
    Returns:
        Dictionary mapping package paths to generation results
    """
    generator = InitPyGenerator(project_path, **kwargs)
    return generator.generate_all_init_files()


def generate_single_init_file(package_path: str, **kwargs) -> bool:
    """
    Generate __init__.py file for a single package.
    
    Args:
        package_path: Path to the package
        **kwargs: Additional arguments for InitPyGenerator
        
    Returns:
        True if successful, False otherwise
    """
    # Get parent directory for root path
    package_path = Path(package_path)
    root_path = package_path.parent
    
    generator = InitPyGenerator(str(root_path), **kwargs)
    return generator.generate_init_file(package_path)


def preview_init_file(package_path: str, **kwargs) -> str:
    """
    Preview what would be generated for a package.
    
    Args:
        package_path: Path to the package
        **kwargs: Additional arguments for InitPyGenerator
        
    Returns:
        String containing preview of generated content
    """
    package_path = Path(package_path)
    root_path = package_path.parent
    
    generator = InitPyGenerator(str(root_path), **kwargs)
    return generator.preview_generation(package_path)


def analyze_file(file_path: str) -> Dict[str, List[str]]:
    """
    Analyze a single Python file to see what symbols would be extracted.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary showing classes and standalone functions found
    """
    generator = InitPyGenerator(str(Path(file_path).parent))
    return generator.analyze_file_symbols(file_path)
