import numpy as np


def generate_ar(alpha, sigma_squared, n, history=None):
    """
    Generate n new points of an autoregressive (AR) time series.

    Parameters:
        alpha (list): AR coefficients.
        sigma_squared (float): Variance of the noise.
        n (int): Number of new points to generate.
        history (array-like, optional): Previous p values of the series.

    Returns:
        tuple:
            new_values (np.ndarray): Generated points.
            new_history (np.ndarray): Last p values of the resulting series.
    """

    p = len(alpha)
    sigma = np.sqrt(sigma_squared)

    # Initialize history if this is the first concept
    if history is None:
        history = np.random.normal(0, sigma, p)
    else:
        history = np.asarray(history, dtype=float).copy()

        if len(history) != p:
            raise ValueError(
                f"History must contain exactly {p} values."
            )

    # Build the new values array
    new_values = np.empty(n)

    for i in range(n):
        noise = np.random.normal(0, sigma)

        new_value = (
            sum(alpha[j] * history[-j - 1] for j in range(p))
            + noise
        )

        new_values[i] = new_value

        # Slide the history window
        history = np.append(history[1:], new_value)

    return new_values, history


def interpolate_parameters(
    alpha_old,
    sigma_squared_old,
    alpha_new,
    sigma_squared_new,
    n
):
    """
    Interpolate AR model parameters between two concepts.

    The AR coefficients and the noise variance are interpolated
    independently using linear interpolation.

    Parameters:
        alpha_old (list): AR coefficients of the old concept.
        sigma_squared_old (float): Noise variance of the old concept.
        alpha_new (list): AR coefficients of the new concept.
        sigma_squared_new (float): Noise variance of the new concept.
        n (int): Number of interpolation steps.

    Returns:
        tuple:
            alpha_interpolated (np.ndarray):
                AR coefficients for each interpolation step.

            sigma_squared_interpolated (np.ndarray):
                Noise variance for each interpolation step.
    """

    alpha_interpolated = np.linspace(
        alpha_old,
        alpha_new,
        n
    )

    sigma_squared_interpolated = np.linspace(
        sigma_squared_old,
        sigma_squared_new,
        n
    )

    return alpha_interpolated, sigma_squared_interpolated


def generate_ar_varying_parameters(
    alphas,
    sigmas_squared,
    history=None
):
    """
    Generate an AR time series with parameters varying over time.

    Parameters:
        alphas (list of lists or np.ndarray):
            AR coefficients for each time step.

        sigmas_squared (list or np.ndarray):
            Noise variance for each time step.

        history (array-like, optional):
            Previous p values of the series.

    Returns:
        tuple:
            new_values (np.ndarray):
                Generated points.

            new_history (np.ndarray):
                Last p values of the resulting series.
    """

    n = len(alphas)
    p = len(alphas[0])

    # The initial history needs an initial noise scale.
    sigma = np.sqrt(sigmas_squared[0])

    # Initialize history if this is the first concept
    if history is None:
        history = np.random.normal(0, sigma, p)
    else:
        history = np.asarray(history, dtype=float).copy()

        if len(history) != p:
            raise ValueError(
                f"History must contain exactly {p} values."
            )

    # Build the new values array
    new_values = np.empty(n)

    for i in range(n):
        sigma = np.sqrt(sigmas_squared[i])

        noise = np.random.normal(0, sigma)

        new_value = (
            sum(
                alphas[i][j] * history[-j - 1]
                for j in range(p)
            )
            + noise
        )

        new_values[i] = new_value

        # Slide the history window
        history = np.append(history[1:], new_value)

    return new_values, history

def generate_gradual_transition(
    alpha_old,
    sigma_squared_old,
    alpha_new,
    sigma_squared_new,
    n,
    history=None
):
    alphas, sigmas_squared = interpolate_parameters(
        alpha_old,
        sigma_squared_old,
        alpha_new,
        sigma_squared_new,
        n
    )

    series, history = generate_ar_varying_parameters(
        alphas,
        sigmas_squared,
        history
    )

    return series, history