# FEDD: Feature Extraction for Explicit Concept Drift Detection in Time Series

An implementation of the **FEDD** (Feature Extraction for Explicit Concept Drift Detection) algorithm for online explicit concept drift detection in time series. This repository is based on the research paper: *"FEDD: Feature Extraction for Explicit Concept Drift Detection in Time Series"* by Rodolfo C. Cavalcante, Leandro L. Minku, and Adriano L. I. Oliveira[cite: 1].

---

## Overview

Time series data streams frequently experience concept drift—changes in the underlying data distribution that negatively impact forecasting and analysis models[cite: 1]. Traditional error-based drift detection methods monitor predictor errors, which can suffer from lag, overfitting, or false alarms[cite: 1]. 

FEDD takes a white-box approach by monitoring statistical features of the time series directly[cite: 1]. By tracking the evolution of pre-defined linear and nonlinear features over a sliding window, FEDD can effectively detect both abrupt and gradual concept drifts with low detection delay and minimal false alarms[cite: 1].

---

## Key Features

- **Online Drift Detection:** Sequentially processes incoming time series observations to identify change points in real time[cite: 1].
- **Feature Extraction (FE) Module:** Automatically computes a rich set of statistical features to capture linear and nonlinear behaviors:
  - **Linear Features:** Autocorrelation (first 5 lags), Partial Autocorrelation (first 5 lags), Variance, Skewness coefficient, Kurtosis coefficient, and Turning points rate[cite: 1].
  - **Nonlinear Features:** Bicorrelation (first 3 lags) and Mutual Information (first 3 lags)[cite: 1].
- **Drift Detection (DD) Module:** Utilizes Exponentially Weighted Moving Average (EWMA) control charts (ECDD) combined with distance metrics to test for significant changes[cite: 1].
- **Distance Metrics Supported:** Supports both **Cosine Distance** and **Pearson Correlation Distance** to compute dissimilarities between feature vectors without scale sensitivity issues[cite: 1].

---

## Architecture & Workflow

FEDD operates through two primary modules:
1. **Feature Extraction Module:** Extracts an initial reference feature vector ($fv_0$) from a baseline window and continuously recomputes feature vectors ($fv_t$) on a sliding window of size $m$[cite: 1].
2. **Drift Detection Module:** Computes the dissimilarity distance ($d_t$) between the current and reference feature vectors, monitoring its EWMA estimator ($Z_t$) against warning ($W$) and drift ($C$) control thresholds[cite: 1].

---

## Getting Started

### Prerequisites

Ensure you have Python installed along with standard scientific and numerical computing libraries (such as `numpy`, `scipy`, etc.).

### Installation

Clone the repository:
```bash
git clone [https://github.com/IvanCEBezerra/Implementation-FEDD-from-Article.git](https://github.com/IvanCEBezerra/Implementation-FEDD-from-Article.git)
cd Implementation-FEDD-from-Article
