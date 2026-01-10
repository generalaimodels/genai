```python 
# ============================================================================
# EXAMPLE USAGE FOR DIFFERENT SCENARIOS
# ============================================================================

EXAMPLE 1: Silent auto-import (recommended for production)
=========================================================

# In your __init__.py:
from .enhanced_dynamic_loader import auto_import_all

# Silent, efficient loading with no warnings
imported_objects, __all__ = auto_import_all(silent=True)
globals().update(imported_objects)

EXAMPLE 2: One-liner injection (simplest approach)
=================================================

# In your __init__.py:
from .enhanced_dynamic_loader import inject_into_init
inject_into_init()  # That's it!

EXAMPLE 3: Configured loading with blacklist
============================================

# In your __init__.py:
from .enhanced_dynamic_loader import configure_loader, auto_import_all

# Configure before loading
configure_loader(
    blacklist=['problematic_module', 'another_module'],
    clear_caches=True
)

imported_objects, __all__ = auto_import_all(silent=True)
globals().update(imported_objects)

EXAMPLE 4: Debug mode with detailed reporting
============================================

# In your __init__.py (for development):
from .enhanced_dynamic_loader import EnhancedEliteDynamicLoader

loader = EnhancedEliteDynamicLoader(silent_mode=False)
imported_objects, __all__ = loader.load_all()
globals().update(imported_objects)

# Optional: Print debug report
report = loader.get_load_report()
print(f"Loaded {report['statistics']['modules_loaded']} modules successfully")

EXAMPLE 5: Whitelist mode (only load specific modules)
=====================================================

# In your __init__.py:
from .enhanced_dynamic_loader import configure_loader, auto_import_all

configure_loader(whitelist=['module1', 'module2', 'safe_module'])
imported_objects, __all__ = auto_import_all(silent=True)
globals().update(imported_objects)


```