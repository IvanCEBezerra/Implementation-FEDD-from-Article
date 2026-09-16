"""Features extraction subpackage - public API (explícita)."""

from .distances import cosine_distance as cosine_distance
from .distances import pearson_distance as pearson_distance
from .features import calculate_bicorrelation as calculate_bicorrelation
from .features import calculate_mutual_information as calculate_mutual_information
from .features import calculate_turning_points as calculate_turning_points
from .features import extract_features as extract_features
from .features import extract_nonlinear_features as extract_nonlinear_features

__all__ = [
    "cosine_distance",
    "pearson_distance",
    "calculate_turning_points",
    "calculate_bicorrelation",
    "calculate_mutual_information",
    "extract_nonlinear_features",
    "extract_features",
]
