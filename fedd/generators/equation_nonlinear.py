import numpy as np

def generate_nonlinear(alpha, sigma_squared, n, history=None):
    """
    Generate a nonlinear time series based on the given parameters.

    Parameters:
        alpha (list of float): Coefficients for the nonlinear model.
        sigma_squared (float): Variance of the noise.
        n (int): Number of samples to generate.
        history (list of float, optional): Initial noise values for the series.

    Returns:
        np.ndarray: Generated nonlinear time series.
    """
    sigma = np.sqrt(sigma_squared)
    if history is None:
        history = np.random.normal(0, sigma, size=2).tolist()  # Initialize with two random values

    series = []
    for i in range(n):
        wt = np.random.normal(0, sigma)
        wt_minus_1 = history[-1]
        wt_minus_2 = history[-2] 
        xt = wt - alpha[0] * wt_minus_1 + alpha[1] * wt_minus_2 + alpha[2] * (wt_minus_1 * wt_minus_2) - alpha[3] * (wt_minus_2 ** 2)
        series.append(xt)
        history = np.array([wt_minus_1, wt])  # Update history with the new wt

    return series, history

def generate_star1(alpha, sigma_squared, n, history=None):
    """
    Generate a time series based on the STAR1 model.

    Parameters:
        alpha (list of float): Coefficients for the STAR1 model.
        sigma_squared (float): Variance of the noise.
        n (int): Number of samples to generate.
        history (list of float, optional): Initial values for the series.

    Returns:
        np.ndarray: Generated STAR1 time series.
    """
    sigma = np.sqrt(sigma_squared)
    if history is None:
        history = np.random.normal(0, sigma, 4).tolist()  # Initialize with four random values

   
    series = []
    for i in range(n):
        wt = np.random.normal(0, sigma)
        transition = -1 / np.expm1(-10 * history[-1])
        ar_part = (
            alpha[0] * history[-1] + 
            alpha[1] * history[-2] + 
            alpha[2] * history[-3] + 
            alpha[3] * history[-4]
        )
        xt = (ar_part * transition) + wt
        series.append(xt)
        history = np.append(history[1:], xt)  # Update history with the new xt

    return series, history

def generate_star2(alpha, sigma_squared, n, history=None):
    """
    Generate a time series based on the STAR2 model.

    Parameters:
        alpha (list of float): Coefficients for the STAR2 model.
        sigma_squared (float): Variance of the noise.
        n (int): Number of samples to generate.
        history (list of float, optional): Initial values for the series.

    Returns:
        np.ndarray: Generated STAR2 time series.
    """
    sigma = np.sqrt(sigma_squared)
    if history is None:
        history = np.random.normal(0, sigma, 4).tolist()  # Initialize with four random values

    series = []
    for i in range(n):
        wt = np.random.normal(0, sigma)
        transition = -1 / np.expm1(-10 * history[-1])
        x_prev = float(history[-1])

        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            exp_term = np.exp(-10 * x_prev)
            denominator = 1 - exp_term
            transition = 1 / denominator

        if not np.isfinite(transition) or abs(transition) > 1e8:
            print("\nPROBLEMA ENCONTRADO")
            print("i =", i)
            print("x_prev =", x_prev)
            print("exp =", exp_term)
            print("denominator =", denominator)
            print("transition =", transition)
            raise FloatingPointError("Transição numericamente problemática")
        ar_part = ( 
            alpha[2] * history[-3] + 
            alpha[0] * history[-2]
        )
        xt = (ar_part * transition) + wt + alpha[0] * history[-1] + alpha[1] * history[-2]
        series.append(xt)
        history = np.append(history[1:], xt)  # Update history with the new xt

    return series, history