# RSS-MoD World Model Architecture
# Recursive State-Space Mixture-of-Depths with Latent JEPA
# SOTA Kernel-Level Implementation

__version__ = "0.1.0"

from .config import RSSMoDConfig
from .model import RSSMoDModel

__all__ = ["RSSMoDConfig", "RSSMoDModel", "__version__"]
