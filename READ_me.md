# 🌦️ Manipal Weather Predictor

This project implements a comprehensive deep learning pipeline to analyze historical weather patterns and forecast future temperature and precipitation for Manipal, Karnataka. The system leverages various Recurrent Neural Network (RNN) architectures to handle time-series data.

## 📊 Dataset Overview
The model is trained on a dataset containing **5,480 daily observations** (spanning approximately 15 years). 

### Key Features:
- **Primary Targets**: `temperature_2m_mean (°C)` and `precipitation_sum (mm)`.
- **Atmospheric Data**: Wind speed/direction, pressure, cloud cover, and dew point.
- **Environmental Metrics**: Shortwave radiation, evapotranspiration, and sunshine duration.
- **Soil Metrics**: Soil moisture (0 to 7cm).

## 🛠️ Project Workflow

### 1. Data Processing
- **Automated Loading**: Fetches weather data directly from Google Drive using `gdown`.
- **Filtering**: Specifically isolates a one-year window (e.g., Jan 2025 – Jan 2026) to analyze current climatic cycles.
- **Normalization**: Features are scaled using a custom `StandardScalerNP` class to ensure the model converges efficiently without data leakage.

### 2. Time-Series Engineering
- **Sliding Window**: Implements a `WINDOW_DAYS = 7` approach, where the model looks at the past week of data to predict the weather for the next day (`HORIZON = 1`).
- **Supervised Learning Format**: Converts raw time-series data into $(samples, window, features)$ tensors.

### 3. Model Architectures
The notebook evaluates and compares several deep learning configurations:
- **Cell Types**: Simple RNN, Gated Recurrent Unit (**GRU**), and Long Short-Term Memory (**LSTM**).
- **Complexity**: Supports **Stacked** layers for depth and **Bidirectional** wrappers to capture patterns from both past and future contexts within the window.
- **Optimization**: Uses Mean Squared Error (**MSE**) as the loss function and the **Adam** optimizer.

## 📈 Evaluation & Results
The project performs a rigorous comparison between two approaches:
1. **Univariate**: Predicting based only on historical temperature and precipitation.
2. **Multivariate**: Using all 13 available numeric features (wind, pressure, etc.).

**Key Metrics calculated:**
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Multivariate Improvement**: Typically, the multivariate model shows significant error reduction by utilizing cross-feature correlations.

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib gdown
