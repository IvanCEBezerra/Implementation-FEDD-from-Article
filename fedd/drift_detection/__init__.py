"""Drift detection subpackage - public API (explícita)."""

# Order: ecdd first, detector second
from .ecdd import ECDDDetector as ECDDDetector
from .detector import FEDDDetector as FEDDDetector
from .elm import ELMRegressor as ELMRegressor

__all__ = [
    "ECDDDetector",
    "FEDDDetector",
    "ELMRegressor"
]
