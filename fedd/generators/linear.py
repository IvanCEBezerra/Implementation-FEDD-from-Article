import numpy as np

from .ar import generate_ar
from .ar import generate_gradual_transition

def generate_linear_abrupt(concepts, n, history=None ):
    """
    Generate a piecewise-linear time series with abrupt drifts.

    Parameters:
        concepts (list of tuples):
            Each tuple contains (alpha, sigma_squared) for a concept.

        n (int):
            Number of samples per concept.
    """
    
    series = []
    for alpha, sigma_squared in concepts:
        series_concept, history = generate_ar(
            alpha,
            sigma_squared,
            n,
            history=history
        )
        series.append(series_concept)

    return np.concatenate(series)

def generate_linear_gradual( concepts, n_per_concept, n_transition, history=None):
    """
    Generate a piecewise-linear time series with gradual drifts.

    Parameters:
        concepts (list of tuples):
            Each tuple contains (alpha, sigma_squared) for a concept.

        n_per_concept (int):
            Number of samples per concept.

        n_transition (int):
            Number of samples in the transition.
    """
    n_stable = n_per_concept - n_transition
    series = []
    for i in range(len(concepts) - 1):
        alpha_old, sigma_squared_old = concepts[i]
        alpha_new, sigma_squared_new = concepts[i + 1]

        series_stable, history = generate_ar(
            alpha_old,
            sigma_squared_old,
            n_stable,
            history=history
        )
        series.append(series_stable)

        series_transition, history = generate_gradual_transition(
            alpha_old,
            sigma_squared_old,
            alpha_new,
            sigma_squared_new,
            n_transition,
            history
        )
        series.append(series_transition)
    # Generate the last stable concept
    alpha_last, sigma_squared_last = concepts[-1]
    series_last, history = generate_ar(
        alpha_last,
        sigma_squared_last,
        n_per_concept,
        history=history)

    series.append(series_last)
    return np.concatenate(series)