# Stock Price Prediction Project

This project implements various machine learning models for stock price prediction.

## Project Structure

```
.
├── src/                    # Source code files
│   ├── analysis_utils.py   # Utility functions for data analysis
│   ├── data_preprocessing.py # Data preprocessing functions
│   ├── lstm_model.py       # LSTM model implementation
│   ├── moirai_transformer.py # Transformer model implementation
│   └── random_forest_model.py # Random Forest model implementation
│
├── plots/                  # Generated plots and visualizations
│   ├── correlation_heatmap.png
│   ├── lstm_*.png         # LSTM model plots
│   ├── moirai_*.png       # Transformer model plots
│   └── rf_*.png           # Random Forest model plots
│
├── results/               # Analysis results and metrics
│   └── analysis_results.json  # Model performance metrics
│
├── models/                # Saved model files
│   ├── lstm_model.h5      # LSTM model weights
│   └── moirai_model.pth   # Transformer model weights
│
├── raw_data/             # Original stock price data
│   ├── AAPL.csv          # Apple stock data
│   ├── GOOGL.csv         # Google stock data
│   └── ...               # Other tech companies
│
├── processed_data/        # Processed dataset files
│   └── AAPL_processed.csv # Processed Apple stock data
│
├── requirements.txt       # Project dependencies (root level for easy installation)
└── README.md             # Project documentation
```

## Models Implemented

1. **Moirai Transformer**: A custom transformer model for time series forecasting
2. **LSTM**: Long Short-Term Memory neural network
3. **Random Forest**: Ensemble learning model

## Setup and Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preprocessing
First, run the data preprocessing script to prepare the data:

```bash
python src/data_preprocessing.py
```

This will:
- Load raw data from `raw_data/`
- Calculate technical indicators
- Save processed data to `processed_data/AAPL_processed.csv`

### 2. Running Models
You can run the models in any order. Each model will automatically load the processed data.

#### Moirai Transformer Model
```bash
python src/moirai_transformer.py
```
Outputs:
- Model file: `models/moirai_model.pth`
- Plots: 
  - `plots/moirai_predictions.png`
  - `plots/moirai_cv_training_history.png`
  - `plots/moirai_error_distribution.png`

#### LSTM Model
```bash
python src/lstm_model.py
```
Outputs:
- Model file: `models/lstm_model.h5`
- Plots:
  - `plots/lstm_predictions.png`
  - `plots/lstm_training_history.png`
  - `plots/lstm_feature_importance.png`

#### Random Forest Model
```bash
python src/random_forest_model.py
```
Outputs:
- Model files:
  - `models/random_forest_model.joblib`
  - `models/random_forest_scaler.joblib`
- Plots:
  - `plots/random_forest_predictions.png`
  - `plots/random_forest_feature_importance.png`

### 3. Analysis Results
After running the models:
- Performance metrics and analysis results are saved in `results/analysis_results.json`
- Correlation analysis is saved as `plots/correlation_heatmap.png`

### Model Performance Comparison

1. **Moirai Transformer**:
   - RMSE: $14.18
   - MAE: $11.22
   - R² Score: 0.8838
   - Direction Accuracy: 52.66%

2. **LSTM**:
   - RMSE: $0.12
   - MAE: $0.10
   - R² Score: -0.9487
   - Direction Accuracy: 48.41%

3. **Random Forest**:
   - RMSE: $8.47
   - MAE: $6.87
   - R² Score: 0.5190
