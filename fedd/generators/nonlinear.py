import numpy as np

from .ar import interpolate_parameters
from .equation_nonlinear import generate_nonlinear
from .equation_nonlinear import generate_star1
from .equation_nonlinear import generate_star2

def generate_nonlinear_series_abrupt(concepts, n_per_concept, model, history=None):
    """
    Generate a nonlinear time series with abrupt changes in concepts.

    Parameters:
        concepts (list of dict): List of concept dictionaries, each containing 'alpha' and 'sigma_squared'.
        n_per_concept (int): Number of samples to generate for each concept.
        model (str): The model to use for generation ('nonlinear', 'star1', or 'star2').
        history (list of float, optional): Initial values for the series.

    Returns:
        np.ndarray: Generated nonlinear time series with abrupt changes.
    """
    series = []
    current_history = history

    for i in range(len(concepts)):
        concept = concepts[i]
        alpha = concept['alpha']
        sigma_squared = concept['sigma_squared']

        if model == 'nonlinear':
            new_series, current_history = generate_nonlinear(alpha, sigma_squared, n_per_concept, current_history)
        elif model == 'star1':
            new_series, current_history = generate_star1(alpha, sigma_squared, n_per_concept, current_history)
        elif model == 'star2':
            new_series, current_history = generate_star2(alpha, sigma_squared, n_per_concept, current_history)
        else:
            raise ValueError(f"Unknown model: {model}")

        series.extend(new_series)

    return np.array(series)

def generate_nonlinear_series_gradual(concepts, n_per_concept, n_transitions, model, history=None):
    """
    Generate a nonlinear time series with gradual changes in concepts.

    Parameters:
        concepts (list of dict): List of concept dictionaries, each containing 'alpha' and 'sigma_squared'.
        n_per_concept (int): Number of samples to generate for each concept.
        n_transitions (int): Number of samples for transitions between concepts.
        model (str): The model to use for generation ('nonlinear', 'star1', or 'star2').
        history (list of float, optional): Initial values for the series.

    Returns:
        np.ndarray: Generated nonlinear time series with gradual changes.
    """
    series = []
    current_history = history
    n_stable = n_per_concept - n_transitions

    for i in range(len(concepts) - 1):
        concept_start = concepts[i]
        concept_end = concepts[i + 1]

        # Generate stable part for the starting concept
        alpha_start = concept_start['alpha']
        sigma_squared_start = concept_start['sigma_squared']
        

        if model == 'nonlinear':
            stable_series, current_history = generate_nonlinear(alpha_start, sigma_squared_start, n_stable, current_history)
        elif model == 'star1':
            stable_series, current_history = generate_star1(alpha_start, sigma_squared_start, n_stable, current_history)
        elif model == 'star2':
            stable_series, current_history = generate_star2(alpha_start, sigma_squared_start, n_stable, current_history)
        else:
            raise ValueError(f"Unknown model: {model}")

        series.extend(stable_series)

        # Generate transition part
        alpha_end = concept_end['alpha']
        sigma_squared_end = concept_end['sigma_squared']

        alphas_interp, sigmas_interp = interpolate_parameters(
            alpha_start, sigma_squared_start, alpha_end, sigma_squared_end, n_transitions
        )

        for j in range(n_transitions):
            # Interpolate parameters
            alpha_interp = alphas_interp[j]
            sigma_squared_interp = sigmas_interp[j]

            if model == 'nonlinear':
                transition_series, current_history = generate_nonlinear(alpha_interp, sigma_squared_interp, 1, current_history)
            elif model == 'star1':
                transition_series, current_history = generate_star1(alpha_interp, sigma_squared_interp, 1, current_history)
            elif model == 'star2':
                transition_series, current_history = generate_star2(alpha_interp, sigma_squared_interp, 1, current_history)
            else:
                raise ValueError(f"Unknown model: {model}")

            series.extend(transition_series)
    # Generate stable part for the last concept
    concept_last = concepts[-1]
    alpha_last = concept_last['alpha']
    sigma_squared_last = concept_last['sigma_squared']

    if model == 'nonlinear':
        stable_series, _ = generate_nonlinear(alpha_last, sigma_squared_last, n_per_concept, current_history)
    elif model == 'star1':
        stable_series, _ = generate_star1(alpha_last, sigma_squared_last, n_per_concept, current_history)
    elif model == 'star2':
        stable_series, _ = generate_star2(alpha_last, sigma_squared_last, n_per_concept, current_history)
    else:
        raise ValueError(f"Unknown model: {model}")

    series.extend(stable_series)

    return np.array(series)