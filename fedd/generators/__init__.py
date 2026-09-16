"""Generators subpackage - public API (explícita, sem mágica dinâmica).

Expõe a API pública via imports explícitos. Evita colisão entre
submódulos e remove dependência de PEP 562 / pkgutil.iter_modules.
"""

# AR base
from .ar import generate_ar as generate_ar
from .ar import generate_ar_varying_parameters as generate_ar_varying_parameters
from .ar import generate_gradual_transition as generate_gradual_transition
from .ar import interpolate_parameters as interpolate_parameters

# Linear
from .linear import generate_linear_abrupt as generate_linear_abrupt
from .linear import generate_linear_gradual as generate_linear_gradual

# Nonlinear
from .equation_nonlinear import generate_nonlinear as generate_nonlinear
from .equation_nonlinear import generate_star1 as generate_star1
from .equation_nonlinear import generate_star2 as generate_star2
from .nonlinear import generate_nonlinear_series_abrupt as generate_nonlinear_series_abrupt
from .nonlinear import generate_nonlinear_series_gradual as generate_nonlinear_series_gradual

# Configs
from .configs import LINEAR_1 as LINEAR_1
from .configs import LINEAR_2 as LINEAR_2
from .configs import LINEAR_3 as LINEAR_3

__all__ = [
    "generate_ar",
    "generate_ar_varying_parameters",
    "generate_gradual_transition",
    "interpolate_parameters",
    "generate_linear_abrupt",
    "generate_linear_gradual",
    "generate_nonlinear",
    "generate_star1",
    "generate_star2",
    "generate_nonlinear_series_abrupt",
    "generate_nonlinear_series_gradual",
    "LINEAR_1",
    "LINEAR_2",
    "LINEAR_3",
]
