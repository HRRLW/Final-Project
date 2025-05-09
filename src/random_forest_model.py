import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from analysis_utils import AnalysisUtils
from typing import Dict, List, Tuple, Any

class StockPricePredictor:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2):
        """
        Initialize the Random Forest model with controlled tree depth to prevent overfitting
        
        Parameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        - min_samples_split: Minimum samples required to split a node
        - min_samples_leaf: Minimum samples required in a leaf node
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_data(self, df, target_col='Close', prediction_days=5):
        """
        Prepare data for training by creating features and target variables
        
        Parameters:
        - df: DataFrame containing the stock data
        - target_col: Column to predict
        - prediction_days: Number of days to predict ahead
        
        Returns:
        - X: Features
        - y: Target values
        """
        # Create the target variable (future price)
        df['Target'] = df[target_col].shift(-prediction_days)
        
        # Create additional features
        df['Price_Change'] = df[target_col].pct_change()
        df['Volatility'] = df[target_col].rolling(window=10).std()
        
        # Select only numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col != 'Target']
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        features_df = df[feature_columns]
        targets = df['Target']
        
        return features_df, targets
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Tune model hyperparameters using GridSearchCV"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        best_params = grid_search.best_params_
        self.model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )
        
        return {
            'best_params': best_params,
            'best_score': np.sqrt(-grid_search.best_score_),
            'cv_results': grid_search.cv_results_
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Train the model using time series cross-validation"""
        # Split data into training and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/random_forest_model.joblib')
        joblib.dump(self.scaler, 'models/random_forest_scaler.joblib')
        
        # Make predictions on validation set
        val_predictions = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_val, val_predictions),
            'rmse': np.sqrt(mean_squared_error(y_val, val_predictions)),
            'mae': mean_absolute_error(y_val, val_predictions),
            'r2': r2_score(y_val, val_predictions)
        }
        
        # Calculate confidence intervals
        confidence_intervals = AnalysisUtils.calculate_confidence_intervals(
            val_predictions, y_val.values)
        metrics.update({
            'confidence_intervals': confidence_intervals
        })
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        AnalysisUtils.plot_feature_importance('Random Forest', feature_importance)
        
        # Save validation results for plotting
        self.y_val = y_val
        self.val_predictions = val_predictions
        
        return metrics, feature_importance
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        - X: Features to predict on
        
        Returns:
        - Predicted values
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def plot_predictions(self, title="Stock Price Predictions"):
        """
        Plot actual vs predicted values from validation set
        
        Parameters:
        - title: Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_val.values, label='Actual', color='blue')
        plt.plot(self.val_predictions, label='Predicted', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/random_forest_predictions.png')
        plt.close()

if __name__ == "__main__":
    # Load and analyze the data
    print("Loading and analyzing data...")
    data = pd.read_csv('processed_data/AAPL_processed.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Analyze dataset
    analysis_results = AnalysisUtils.analyze_dataset(data)
    print("\nDataset Analysis:")
    print(f"Time Range: {analysis_results['time_range']['start']} to {analysis_results['time_range']['end']}")
    print(f"Trading Days: {analysis_results['time_range']['trading_days']}")
    print("\nMissing Values:")
    for col, count in analysis_results['missing_values'].items():
        if count > 0:
            print(f"{col}: {count}")
    
    # Plot correlation heatmap
    print("\nGenerating correlation heatmap...")
    AnalysisUtils.plot_correlation_heatmap(data)
    print("Correlation heatmap has been saved as 'correlation_heatmap.png'")
    
    # Initialize predictor
    print("\nInitializing Random Forest model...")
    predictor = StockPricePredictor()
    
    # Prepare the data
    print("Preparing data...")
    X, y = predictor.prepare_data(data)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Tune hyperparameters
    print("\nTuning hyperparameters (this may take a while)...")
    tuning_results = predictor.tune_hyperparameters(X, y)
    print("\nHyperparameter Tuning Results:")
    print("Best Parameters:")
    for param, value in tuning_results['best_params'].items():
        print(f"{param}: {value}")
    print(f"Best Cross-validation RMSE: ${tuning_results['best_score']:.2f}")
    
    # Train the model with best parameters
    print("\nTraining model with best parameters...")
    metrics, feature_importance = predictor.train(X, y)
    
    # Print comprehensive results
    print("\nModel Performance:")
    print(f"Root Mean Squared Error: ${metrics['rmse']:.2f}")
    print(f"Mean Absolute Error: ${metrics['mae']:.2f}")
    print(f"R-squared Score: {metrics['r2']:.4f}")
    
    # Print confidence intervals
    ci = metrics['confidence_intervals']
    print(f"\nConfidence Intervals ({ci['confidence_level']*100}%):")
    print(f"Mean Error: ${ci['mean_error']:.2f}")
    print(f"Interval: (${ci['confidence_interval'][0]:.2f}, ${ci['confidence_interval'][1]:.2f})")
    
    # Print feature importance
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string())
    
    # Plot results
    print("\nGenerating visualizations...")
    predictor.plot_predictions("Random Forest Stock Price Predictions (Validation Set)")
    print("Generated visualization files:")
    print("- random_forest_predictions.png")
    print("- random_forest_feature_importance.png")
    print("\nAnalysis complete!")
