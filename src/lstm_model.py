import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import seaborn as sns

class LSTMPredictor:
    def __init__(self, sequence_length=10, lstm_units=128, dropout_rate=0.3):
        """
        Initialize LSTM model for stock price prediction
        
        Parameters:
        - sequence_length: Number of time steps to look back
        - lstm_units: Number of LSTM units in each layer
        - dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data):
        """
        Create sequences for LSTM input
        
        Parameters:
        - data: Input data array
        
        Returns:
        - X: Sequences of features
        - y: Target values
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, target_col='Close'):
        """
        Prepare data for LSTM model with enhanced features
        
        Parameters:
        - df: DataFrame containing the stock data
        - target_col: Column to predict
        
        Returns:
        - Processed X and y data for training
        """
        # Technical indicators
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Additional momentum indicators
        df['MOM'] = df['Close'].diff(10)
        df['ROC'] = df['Close'].pct_change(10) * 100
        
        # Volume indicators
        df['OBV'] = (df['Close'].diff() > 0).astype(int) * df['Volume']
        df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']
        
        # Select features
        feature_columns = [
            # Price data
            'Open', 'High', 'Low', 'Close', 'Volume',
            # Moving averages
            'MA5', 'MA20', 'EMA12', 'EMA26',
            # Technical indicators
            'RSI', 'MACD', 'Signal_Line', 'MACD_Hist',
            'BB_Width', 'MOM', 'ROC',
            # Volume indicators
            'OBV', 'VWAP',
            # Market conditions
            'Price_Change', 'Volatility',
            # Economic indicators
            'Unemployment_Rate', 'CPI', 'Fed_Funds_Rate', 'SP500'
        ]
        
        # Remove any NaN values
        df = df.dropna()
        
        # Scale the features
        self.feature_columns = feature_columns  # Store for later use
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Get the target column index
        target_idx = feature_columns.index(target_col)
        
        # Use only the target column for y
        y = y[:, target_idx]
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build an advanced LSTM model with attention mechanism
        
        Parameters:
        - input_shape: Shape of input data (sequence_length, n_features)
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        x = LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = Dropout(self.dropout_rate)(x)
        
        # Second LSTM layer with increased units
        x = LSTM(self.lstm_units * 2, return_sequences=True)(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        
        # Third LSTM layer
        x = LSTM(self.lstm_units)(attention)
        x = Dropout(self.dropout_rate)(x)
        
        # Dense layers for better feature extraction
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom learning rate schedule
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # Huber loss for robustness
            metrics=['mae', 'mse']
        )
        
        return self.model
    
    def train(self, X, y, validation_split=0.2, epochs=200, batch_size=64):
        """
        Train the LSTM model
        
        Parameters:
        - X: Training features
        - y: Target values
        - validation_split: Proportion of data for validation
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        
        Returns:
        - Training history
        """
        # Build the model
        self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                mode='min'
            ),
            ModelCheckpoint(
                'models/lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                mode='min'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate the model's performance on test data
        
        Parameters:
        - X_test: Test features
        - y_test: True test values
        
        Returns:
        - Dictionary containing various performance metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        y_direction = np.sign(np.diff(y_test))
        pred_direction = np.sign(np.diff(y_pred.flatten()))
        directional_accuracy = np.mean(y_direction == pred_direction)
        
        # Calculate confidence intervals
        residuals = y_test - y_pred.flatten()
        confidence_level = 0.95
        confidence_interval = np.percentile(residuals, [(1 - confidence_level) * 100/2, (1 + confidence_level) * 100/2])
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'confidence_intervals': {
                'confidence_level': confidence_level,
                'lower_bound': confidence_interval[0],
                'upper_bound': confidence_interval[1],
                'mean_error': np.mean(residuals)
            }
        }
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        - X: Input features
        
        Returns:
        - Predicted values
        """
        return self.model.predict(X)
    
    def plot_training_history(self, history):
        """
        Plot enhanced training history with multiple metrics
        
        Parameters:
        - history: Training history from model.fit()
        """
        metrics = ['loss', 'mae', 'mse']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LSTM Model Training History', fontsize=16)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            ax.plot(history.history[metric], label=f'Training {metric.upper()}')
            ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
            ax.set_title(f'Model {metric.upper()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True)
        
        # Plot learning rate
        if 'lr' in history.history:
            ax = axes[1, 1]
            ax.plot(history.history['lr'], label='Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/lstm_training_history.png')
        plt.close()
    
    def plot_predictions(self, y_true, y_pred, dates=None, title="Stock Price Predictions"):
        """
        Plot enhanced predictions visualization
        
        Parameters:
        - y_true: Actual values
        - y_pred: Predicted values
        - dates: Optional dates for x-axis
        - title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
        fig.suptitle(title, fontsize=16)
        
        # Main predictions plot
        x = dates if dates is not None else np.arange(len(y_true))
        ax1.plot(x, y_true, label='Actual', color='blue', linewidth=2)
        ax1.plot(x, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
        
        # Add confidence intervals if available
        if hasattr(self, 'confidence_intervals'):
            ci = self.confidence_intervals
            ax1.fill_between(x,
                           y_pred.flatten() + ci['lower_bound'],
                           y_pred.flatten() + ci['upper_bound'],
                           color='gray', alpha=0.2,
                           label=f'{ci["confidence_level"]*100}% Confidence Interval')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Error plot
        errors = y_true - y_pred.flatten()
        ax2.plot(x, errors, color='green', label='Prediction Error')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.fill_between(x, 0, errors, alpha=0.3, color='green')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Error ($)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/lstm_predictions.png')
        plt.close()
        
    def plot_feature_importance(self, X_test):
        """
        Plot feature importance using integrated gradients
        
        Parameters:
        - X_test: Test features to analyze
        """
        # Calculate feature importance using gradient-based method
        test_pred = self.model.predict(X_test)
        gradients = np.zeros_like(X_test)
        
        # Calculate gradients for each feature
        for i in range(X_test.shape[-1]):
            temp_X = X_test.copy()
            temp_X[:, :, i] = 0
            temp_pred = self.model.predict(temp_X)
            gradients[:, :, i] = np.abs(test_pred - temp_pred).mean()
        
        # Average importance across time steps
        feature_importance = gradients.mean(axis=(0, 1))
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Top 15 Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('plots/lstm_feature_importance.png')
        plt.close()
        
        return importance_df

# Define feature columns globally
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                 'MA5', 'MA20', 'RSI', 'Price_Change', 'Volatility',
                 'Unemployment_Rate', 'CPI', 'Fed_Funds_Rate', 'SP500']

if __name__ == "__main__":
    # Load the processed data
    data = pd.read_csv('processed_data/AAPL_processed.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Create Price_Change and Volatility features
    data['Price_Change'] = data['Close'].pct_change()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data = data.dropna()
    
    # Initialize predictor with enhanced parameters
    print("\nInitializing LSTM model...")
    predictor = LSTMPredictor(
        sequence_length=20,  # Increased sequence length
        lstm_units=128,     # Larger LSTM units
        dropout_rate=0.3    # Increased dropout for better regularization
    )
    
    print("Preparing data with enhanced features...")
    X, y = predictor.prepare_data(data)
    print(f"Dataset shape: {X.shape}")
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train the model
    print("\nTraining LSTM model...")
    history = predictor.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = predictor.evaluate_model(X_test, y_test)
    
    print("\nModel Performance Metrics:")
    print(f"Root Mean Squared Error: ${metrics['rmse']:.2f}")
    print(f"Mean Absolute Error: ${metrics['mae']:.2f}")
    print(f"R-squared Score: {metrics['r2']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']*100:.2f}%")
    
    ci = metrics['confidence_intervals']
    print(f"\n{ci['confidence_level']*100}% Confidence Intervals:")
    print(f"Mean Error: ${ci['mean_error']:.2f}")
    print(f"Interval: (${ci['lower_bound']:.2f}, ${ci['upper_bound']:.2f})")
    
    # Store confidence intervals for plotting
    predictor.confidence_intervals = ci
    
    print("\nGenerating visualizations...")
    
    # Plot training history
    predictor.plot_training_history(history)
    print("Training history plot saved as 'lstm_training_history.png'")
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Plot predictions with dates
    test_dates = data['Date'].iloc[-len(y_test):].values
    predictor.plot_predictions(y_test, predictions, dates=test_dates,
                             title="LSTM Stock Price Predictions")
    print("Predictions plot saved as 'lstm_predictions.png'")
    
    # Plot feature importance
    importance_df = predictor.plot_feature_importance(X_test)
    print("\nFeature importance plot saved as 'lstm_feature_importance.png'")
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string())
    
    print("\nAnalysis complete!")
    print("\nTraining plots have been saved as 'lstm_training_history.png'")
    print("Prediction plot has been saved as 'lstm_predictions.png'")
