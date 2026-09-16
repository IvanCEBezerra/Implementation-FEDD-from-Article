"""
Configuration of concepts for time series based on the article:
FEDD: Feature Extraction for Explicit Concept Drift Detection in Time Series
"""

# Linear 1: AR(4) model parameters and sigma_squared
LINEAR_1 = [
    ([0.9, -0.2, 0.8, -0.5], 0.5),
    ([-0.3, 1.4, 0.4, -0.5], 1.5),
    ([1.5, -0.4, -0.3, 0.2], 2.5),
    ([-0.1, 1.4, 0.4, -0.7], 3.5),
]

# Linear 2: AR(6) model parameters and sigma_squared
LINEAR_2 = [
    ([1.1, -0.6, 0.8, -0.5, -0.1, 0.3], 0.5),
    ([-0.1, 1.2, 0.4, 0.3, -0.2, -0.6], 1.5),
    ([1.2, -0.4, -0.3, 0.7, -0.6, 0.4], 2.5),
    ([-0.1, 1.1, 0.5, 0.2, -0.2, -0.5], 3.5),
]

# Linear 3: AR(p) model parameters and sigma_squared (varying order p)
# The smaller arrays were filled with zeros (zero-padding) to match the maximum order p=7
LINEAR_3 = [
    ([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ([1.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], 1.5), # Corrected from article: -0.5 not 0.5
    ([0.9, -0.2, 0.8, -0.5, 0.0, 0.0, 0.0], 2.5),
    ([0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4], 3.5),
]