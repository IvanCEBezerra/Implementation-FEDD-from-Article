# FEDD: Feature Extraction for Explicit Concept Drift Detection in Time Series

An implementation of the **FEDD** (Feature Extraction for Explicit Concept Drift Detection) algorithm for online explicit concept drift detection in time series[cite: 1]. This repository is based on the research paper: *"FEDD: Feature Extraction for Explicit Concept Drift Detection in Time Series"* by Rodolfo C. Cavalcante, Leandro L. Minku, and Adriano L. I. Oliveira[cite: 1].

---

## Overview

Time series data streams frequently experience concept drift—changes in the underlying data distribution that negatively impact forecasting and analysis models[cite: 1]. Traditional error-based drift detection methods monitor predictor errors, which can suffer from lag, overfitting, or high false alarm rates[cite: 1]. 

FEDD takes a white-box, feature-based approach by monitoring statistical features of the time series directly[cite: 1]. By tracking the evolution of pre-defined linear and nonlinear features over a sliding window, FEDD effectively detects both abrupt and gradual concept drifts with low detection delay and minimal false alarms[cite: 1].

---

## Key Features

- **Online Drift Detection (`FEDDDetector`):** Sequentially processes incoming time series observations point-by-point to identify change points in real time[cite: 1, 14].
- **Comprehensive Feature Extraction (20 Dimensions):** Automatically computes a robust feature vector combining[cite: 1, 12]:
  - **Basic & Distribution Statistics:** Turning points rate (calculated on the original series), variance, skewness, and kurtosis (calculated on the differenced series)[cite: 1, 12].
  - **Linear Features:** Autocorrelation (first 5 lags) and Partial Autocorrelation (first 5 lags) on differenced data[cite: 1, 12].
  - **Nonlinear Features:** Bicorrelation (first 3 lags) and Mutual Information (first 3 lags) using `scikit-learn` and custom math[cite: 1, 12].
- **Distance Metrics (`distances.py`):** Supports both **Cosine Distance** and **Pearson Correlation Distance** to compute dissimilarities between feature vectors without scale sensitivity issues[cite: 1, 11].
- **Drift Monitoring (`ECDDDetector`):** Implements an Exponentially Weighted Moving Average (EWMA) control chart mechanism equipped with warning ($W$) and drift ($C$) control thresholds, plus burn-in variance stabilization[cite: 1, 15].
- **Benchmark Baselines (`ELM_ECDD_Detector`):** Includes an analytical Extreme Learning Machine (ELM) regressor and an error-based ELM-ECDD concept drift detection baseline for comparative evaluations[cite: 5, 16, 17].
- **Synthetic Data Generators:** Built-in modular generators to recreate linear (AR models) and nonlinear (NLMA, STAR1, STAR2) artificial benchmark time series datasets with controlled abrupt and gradual concept drifts[cite: 3, 5, 8, 9].

---

## Repository Structure

```text
├── fedd/
│   ├── generators/
│   │   ├── __init__.py                # Public API for data generators[cite: 2]
│   │   ├── ar.py                      # Autoregressive series generation & interpolation[cite: 3]
│   │   ├── configs.py                 # Linear concept parameter setups (LINEAR_1, 2, 3)[cite: 4]
│   │   ├── equation_nonlinear.py      # Nonlinear model equations: NLMA (Eq. 1), STAR1 (Eq. 2), STAR2 (Eq. 3)[cite: 5]
│   │   ├── linear.py                  # Linear abrupt & gradual sequence builders[cite: 8]
│   │   ├── nonlinear.py               # Nonlinear abrupt & gradual sequence builders[cite: 9]
│   │   ├── gen_dataset_linear.py      # Script to generate & save linear benchmark data (.npy)[cite: 6]
│   │   └── gen_dataset_nonlinear.py   # Script to generate & save nonlinear benchmark data (.npy)[cite: 7]
│   ├── features_extration/
│   │   ├── __init__.py                # Public API for feature extraction & distances[cite: 10]
│   │   ├── distances.py               # Cosine and Pearson distance metric implementations[cite: 11]
│   │   └── features.py                # 20-feature extraction pipeline (Turning points, ACF, PACF, BIC, MI)
