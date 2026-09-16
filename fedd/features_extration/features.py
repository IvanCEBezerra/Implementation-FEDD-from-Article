import numpy as np

from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf, pacf


# Turning points

def calculate_turning_points(window):
    """
    Calcula a taxa de turning points da janela.

    Um ponto é considerado turning point quando é:
        - maior que os dois vizinhos; ou
        - menor que os dois vizinhos.

    Esta feature é calculada na série original,
    não na série diferenciada.
    """

    window = np.asarray(window, dtype=float)

    if len(window) < 3:
        return 0.0

    previous = window[:-2]
    current = window[1:-1]
    next_value = window[2:]

    local_maximum = (
        (current > previous)
        & (current > next_value)
    )

    local_minimum = (
        (current < previous)
        & (current < next_value)
    )

    turning_points = (
        local_maximum
        | local_minimum
    )

    return np.mean(turning_points)


# Autocorrelation

def calculate_autocorrelation(diff_window, max_lag=5):
    """
    Calcula as autocorrelações dos lags 1 até max_lag.

    A feature é calculada sobre a série diferenciada.
    """

    diff_window = np.asarray(diff_window, dtype=float)

    if len(diff_window) <= max_lag:
        return np.zeros(max_lag)

    values = acf(
        diff_window,
        nlags=max_lag,
        fft=False
    )

    values = values[1:max_lag + 1]

    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# Partial autocorrelation

def calculate_partial_autocorrelation(diff_window, max_lag=5):
    """
    Calcula as autocorrelações parciais dos lags
    1 até max_lag.

    A feature é calculada sobre a série diferenciada.
    """

    diff_window = np.asarray(diff_window, dtype=float)

    if len(diff_window) <= max_lag:
        return np.zeros(max_lag)

    values = pacf(
        diff_window,
        nlags=max_lag,
        method="yw"
    )

    values = values[1:max_lag + 1]

    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# Bicorrelation

def calculate_bicorrelation(diff_window, max_lag=3):
    """
    Calcula a bicorrelação / three-point autocorrelation
    para os lags 1 até max_lag.

    Para cada lag k:

        E[x_t * x_(t-k) * x_(t-2k)]

    A feature é calculada sobre a série diferenciada.

    Observação:
    o artigo descreve esta feature como bicorrelation /
    three-point autocorrelation, mas não apresenta a
    fórmula no texto devido à limitação de espaço.
    """

    diff_window = np.asarray(diff_window, dtype=float)

    values = []

    for lag in range(1, max_lag + 1):

        if len(diff_window) <= 2 * lag:
            values.append(0.0)
            continue

        x_t = diff_window[2 * lag:]
        x_t_lag = diff_window[lag:-lag]
        x_t_2lag = diff_window[:-2 * lag]

        bicorrelation = np.mean(
            x_t
            * x_t_lag
            * x_t_2lag
        )

        values.append(bicorrelation)

    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# Mutual information

def calculate_mutual_information(diff_window, max_lag=3):
    """
    Calcula a informação mútua entre:

        x_t

    e

        x_(t-k)

    para k = 1, 2, 3.

    A feature é calculada sobre a série diferenciada.

    mutual_info_regression é usada para estimar a MI
    entre as duas variáveis contínuas.
    """

    diff_window = np.asarray(diff_window, dtype=float)

    values = []

    for lag in range(1, max_lag + 1):

        if len(diff_window) <= lag:
            values.append(0.0)
            continue

        x_lagged = diff_window[:-lag]
        x_current = diff_window[lag:]

        # Skip if variable is constant
        if (
            np.all(x_lagged == x_lagged[0])
            or np.all(x_current == x_current[0])
        ):
            values.append(0.0)
            continue

        try:

            mi = mutual_info_regression(
                x_lagged.reshape(-1, 1),
                x_current,
                random_state=42
            )[0]

        except ValueError:
            mi = 0.0

        values.append(mi)

    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# Nonlinear features

def extract_nonlinear_features(diff_window):
    """
    Extrai as duas categorias não lineares:

        - bicorrelation: 3 features
        - mutual information: 3 features

    Total: 6 features.
    """

    bicorrelation = calculate_bicorrelation(
        diff_window,
        max_lag=3
    )

    mutual_information = calculate_mutual_information(
        diff_window,
        max_lag=3
    )

    return (
        bicorrelation,
        mutual_information
    )


# Main feature extraction

def extract_features(window):
    """
    Extrai o vetor completo de características utilizado
    pelo FEDD.

    Estrutura:

        1  - turning points rate
        1  - variance
        1  - skewness
        1  - kurtosis
        5  - autocorrelation
        5  - partial autocorrelation
        3  - bicorrelation
        3  - mutual information

    Total = 20 features.

    De acordo com o artigo, todas as features, exceto
    turning points rate, são calculadas sobre a série
    diferenciada.
    """

    window = np.asarray(
        window,
        dtype=float
    )

    if len(window) < 3:
        raise ValueError(
            "A janela precisa possuir pelo menos 3 pontos."
        )

    # Turning points
    turning_points = calculate_turning_points(
        window
    )

    # Differencing
    diff_window = np.diff(window)

    # Linear features

    autocorrelation = calculate_autocorrelation(
        diff_window,
        max_lag=5
    )

    partial_autocorrelation = (
        calculate_partial_autocorrelation(
            diff_window,
            max_lag=5
        )
    )

    variance = np.var(
        diff_window
    )

    skewness = skew(
        diff_window
    )

    kurt = kurtosis(
        diff_window
    )

    # Nonlinear features
    bicorrelation, mutual_information = (
        extract_nonlinear_features(
            diff_window
        )
    )

    # Final feature vector

    features = np.concatenate([
        np.array([
            turning_points,
            variance,
            skewness,
            kurt
        ]),

        autocorrelation,

        partial_autocorrelation,

        bicorrelation,

        mutual_information
    ])

    # Handle NaN and inf
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features