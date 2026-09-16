"""FEDD - Generators for AR time series with drifts.

API explícita: re-exporta nomes públicos dos subpacotes sem mágica
dinâmica (PEP 562). Evita colisão silenciosa entre pastas — cada nome
é importado explicitamente e conflitos seriam erro de import visível.

Também mantém alias de compatibilidade para imports legados sem prefixo
`fedd.` (ex: `from features_extration.distances import cosine_distance`)
apenas via sys.modules, sem lógica dinâmica.
"""

import sys

__version__ = "0.1.0"

# Re-export subpackages
from fedd.generators import LINEAR_1 as LINEAR_1
from fedd.generators import LINEAR_2 as LINEAR_2
from fedd.generators import LINEAR_3 as LINEAR_3
from fedd.generators import generate_ar as generate_ar
from fedd.generators import generate_ar_varying_parameters as generate_ar_varying_parameters
from fedd.generators import generate_gradual_transition as generate_gradual_transition
from fedd.generators import generate_linear_abrupt as generate_linear_abrupt
from fedd.generators import generate_linear_gradual as generate_linear_gradual
from fedd.generators import generate_nonlinear as generate_nonlinear
from fedd.generators import generate_nonlinear_series_abrupt as generate_nonlinear_series_abrupt
from fedd.generators import generate_nonlinear_series_gradual as generate_nonlinear_series_gradual
from fedd.generators import generate_star1 as generate_star1
from fedd.generators import generate_star2 as generate_star2
from fedd.generators import interpolate_parameters as interpolate_parameters
from fedd.features_extration import calculate_bicorrelation as calculate_bicorrelation
from fedd.features_extration import calculate_mutual_information as calculate_mutual_information
from fedd.features_extration import calculate_turning_points as calculate_turning_points
from fedd.features_extration import cosine_distance as cosine_distance
from fedd.features_extration import extract_features as extract_features
from fedd.features_extration import extract_nonlinear_features as extract_nonlinear_features
from fedd.features_extration import pearson_distance as pearson_distance
from fedd.drift_detection import ECDDDetector as ECDDDetector
from fedd.drift_detection import FEDDDetector as FEDDDetector

__all__ = [
    "__version__",
    "ECDDDetector",
    "FEDDDetector",
    "calculate_bicorrelation",
    "calculate_mutual_information",
    "calculate_turning_points",
    "cosine_distance",
    "extract_features",
    "extract_nonlinear_features",
    "pearson_distance",
    "LINEAR_1",
    "LINEAR_2",
    "LINEAR_3",
    "generate_ar",
    "generate_ar_varying_parameters",
    "generate_gradual_transition",
    "generate_linear_abrupt",
    "generate_linear_gradual",
    "generate_nonlinear",
    "generate_nonlinear_series_abrupt",
    "generate_nonlinear_series_gradual",
    "generate_star1",
    "generate_star2",
    "interpolate_parameters",
]

# Legacy alias for imports without fedd prefix
def _install_legacy_aliases():
    import importlib

    for fedd_name, legacy_name in (
        ("fedd.features_extration", "features_extration"),
        ("fedd.drift_detection", "drift_detection"),
        ("fedd.generators", "generators"),
    ):
        try:
            real_mod = importlib.import_module(fedd_name)
            sys.modules.setdefault(legacy_name, real_mod)
        except Exception:
            continue


_install_legacy_aliases()
