```python

# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Basic lazy module with simple structure
    print("=== Example 1: Basic Lazy Module ===")
    
    basic_structure = {
        "core": ["CoreClass", "core_function"],
        "utils": ["UtilityClass", "helper_function"],
        "advanced": ["AdvancedModel", "Optimizer"]
    }
    
    basic_lazy = AdvancedLazyModule(
        name="example_package",
        module_file=__file__,
        import_structure=basic_structure,
        import_strategy=ImportStrategy.LAZY
    )
    
    print(f"Available attributes: {basic_lazy.__all__}")
    print(f"Module directory: {dir(basic_lazy)[:5]}...")  # Show first 5
    
    # Example 2: Nested module structure with performance monitoring
    print("\n=== Example 2: Nested Module Structure ===")
    
    nested_structure = {
        "ml": {
            "models": {
                "classification": ["RandomForest", "SVM", "NeuralNet"],
                "regression": ["LinearRegression", "PolynomialRegression"],
                "clustering": ["KMeans", "DBSCAN"]
            },
            "preprocessing": ["StandardScaler", "Normalizer"],
            "metrics": ["accuracy_score", "precision_recall"]
        },
        "data": {
            "loaders": ["CSVLoader", "JSONLoader", "DatabaseLoader"],
            "transformers": ["DataTransformer", "FeatureEncoder"]
        },
        "visualization": ["PlotManager", "ChartGenerator"]
    }
    
    advanced_lazy = AdvancedLazyModule(
        name="ml_package",
        module_file=__file__,
        import_structure=nested_structure,
        import_strategy=ImportStrategy.HYBRID,
        cache_strategy=CacheStrategy.WEAK_REF,
        enable_monitoring=True
    )
    
    print(f"Nested modules available: {len(advanced_lazy._modules)}")
    print(f"Total classes/functions: {len(advanced_lazy._class_to_module)}")
    
    # Example 3: Performance monitoring and statistics
    print("\n=== Example 3: Performance Monitoring ===")
    
    # Get initial stats
    stats = advanced_lazy.get_performance_stats()
    print(f"Initial stats: {stats}")
    
    # Simulate some access (would normally trigger real imports)
    try:
        # These would normally import real modules
        # advanced_lazy.ml  # Would create sub-module
        # advanced_lazy.PlotManager  # Would import from visualization module
        pass
    except Exception as e:
        print(f"Expected import error in demo: {type(e).__name__}")
    
    # Example 4: Caching and memory optimization
    print("\n=== Example 4: Cache Management ===")
    
    print(f"Cache strategy: {advanced_lazy._cache_strategy}")
    print("Demonstrating cache operations...")
    
    # Add some test objects to cache
    advanced_lazy._cache.set("test_obj1", "value1")
    advanced_lazy._cache.set("test_obj2", "value2")
    
    # Retrieve from cache
    cached_val = advanced_lazy._cache.get("test_obj1")
    print(f"Retrieved from cache: {cached_val}")
    
    # Clear cache
    advanced_lazy.clear_cache()
    print("Cache cleared")
    
    # Example 5: Error handling and robustness
    print("\n=== Example 5: Error Handling ===")
    
    try:
        # Try to access non-existent attribute
        non_existent = advanced_lazy.non_existent_module
    except AttributeError as e:
        print(f"Handled missing attribute: {e}")
    
    try:
        # Try to access invalid nested structure
        invalid = advanced_lazy.invalid.nested.path
    except AttributeError as e:
        print(f"Handled invalid nested access: {e}")
    
    # Example 6: Integration with TYPE_CHECKING pattern
    print("\n=== Example 6: TYPE_CHECKING Integration ===")
    
    example_type_checking_code = '''
# This is how you would typically use it in a real module:

from typing import TYPE_CHECKING

_import_structure = {
    "core": {
        "base": ["BaseEnvironment", "BaseAgent"],
        "advanced": ["AdvancedEnvironment", "NeuralAgent"]
    },
    "utils": ["Logger", "ConfigManager"],
    "extensions": {
        "plugins": ["PluginA", "PluginB"],
        "addons": ["AddonX", "AddonY"]
    }
}

if TYPE_CHECKING:
    # Import for type checking only
    from .core.base import BaseEnvironment, BaseAgent
    from .core.advanced import AdvancedEnvironment, NeuralAgent
    from .utils import Logger, ConfigManager
    from .extensions.plugins import PluginA, PluginB
    from .extensions.addons import AddonX, AddonY
else:
    # Use lazy loading at runtime
    import sys
    sys.modules[__name__] = AdvancedLazyModule(
        name=__name__,
        module_file=globals()["__file__"],
        import_structure=_import_structure,
        module_spec=__spec__,
        import_strategy=ImportStrategy.HYBRID,
        cache_strategy=CacheStrategy.WEAK_REF,
        enable_monitoring=True
    )
'''
    
    print("Example TYPE_CHECKING integration code structure shown above")
    
    print("\n=== Advanced Features Summary ===")
    print("✓ Recursive sub-module support")
    print("✓ Multiple import strategies (LAZY, EAGER, HYBRID, CACHED)")
    print("✓ Advanced caching with weak references")
    print("✓ Thread-safe operations")
    print("✓ Circular dependency detection")
    print("✓ Performance monitoring and statistics")
    print("✓ Memory optimization")
    print("✓ Comprehensive error handling")
    print("✓ TYPE_CHECKING integration")
    print("✓ Real-time performance optimization")
    print("✓ Scalable architecture for large projects")

```