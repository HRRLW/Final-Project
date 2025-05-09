import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Positional encoding using sine and cosine functions
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor"""
        return x + self.pe[:, :x.size(1)]

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Masked Multi-Head Attention module
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear transformations and reshape
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output)

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Transformer Encoder Layer
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.self_attention = MaskedMultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class MoiraiTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, num_heads: int = 16,
                 num_layers: int = 6, d_ff: int = 512, dropout: float = 0.2):
        """
        Moirai Transformer model for time series forecasting
        
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for self-attention"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input embedding and positional encoding
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Generate mask
        mask = self.generate_mask(x.size(1)).to(x.device)
        
        # Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Output projection
        return self.output_layer(x)

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int):
        """
        Dataset for time series data
        
        Args:
            data: Input data array
            sequence_length: Length of input sequences
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, 0]  # Predict Close price
        return x, y

class MoiraiPredictor:
    def __init__(self, sequence_length: int = 10, batch_size: int = 32,
                 learning_rate: float = 0.001, num_epochs: int = 100):
        """
        Moirai Transformer predictor wrapper
        
        Args:
            sequence_length: Length of input sequences
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        feature_columns = [
            # Price and Volume
            'Open', 'High', 'Low', 'Close', 'Volume',
            # Moving Averages
            'MA5', 'MA20', 'EMA12', 'EMA26',
            # Technical Indicators
            'RSI', 'MACD', 'Signal_Line', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
            'MOM', 'ROC', 'OBV', 'VWAP',
            # Price Derivatives
            'Price_Change', 'Volatility',
            # Economic Indicators
            'Unemployment_Rate', 'CPI', 'Fed_Funds_Rate', 'SP500'
        ]
        
        data = self.scaler.fit_transform(df[feature_columns])
        return data
        
    def time_series_cv_split(self, n_samples: int, n_splits: int = 5) -> list:
        """Generate time series cross-validation splits"""
        splits = []
        split_size = n_samples // (n_splits + 1)
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            val_end = train_end + split_size
            splits.append({
                'train_idx': range(0, train_end),
                'val_idx': range(train_end, min(val_end, n_samples))
            })
        return splits

    def train(self, X: np.ndarray, early_stopping_patience: int = 10):
        """Train the model with time series cross-validation"""
        dataset = TimeSeriesDataset(X, self.sequence_length)
        cv_splits = self.time_series_cv_split(len(dataset), n_splits=5)
        
        # Track metrics across folds
        all_train_losses = []
        all_val_losses = []
        all_val_metrics = []
        
        print("Starting Time Series Cross-Validation...")
        
        for fold, split in enumerate(cv_splits, 1):
            print(f"\nFold {fold}/5")
            
            # Create train and validation datasets for this fold
            train_dataset = torch.utils.data.Subset(dataset, split['train_idx'])
            val_dataset = torch.utils.data.Subset(dataset, split['val_idx'])
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize model
        self.model = MoiraiTransformer(input_dim=X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output[:, -1, 0], batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output = self.model(batch_x)
                    val_loss += criterion(output[:, -1, 0], batch_y).item()
                    val_predictions.extend(output[:, -1, 0].cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'models/moirai_model.pth')
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Early stopping
            if no_improve_count >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Fold {fold}, Epoch [{epoch+1}/{self.num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}')
        
            # Store losses for this fold
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            
            # Calculate metrics on validation set
            self.model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    output = self.model(batch_x)
                    val_predictions.extend(output[:, -1, 0].cpu().numpy())
                    val_targets.extend(batch_y.numpy())
            
            # Convert predictions back to original scale
            dummy_array = np.zeros((len(val_predictions), X.shape[1]))
            dummy_array[:, 0] = val_predictions
            val_predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
            
            dummy_array[:, 0] = val_targets
            val_targets = self.scaler.inverse_transform(dummy_array)[:, 0]
            
            # Calculate metrics
            fold_rmse = np.sqrt(np.mean((val_targets - val_predictions) ** 2))
            fold_mae = np.mean(np.abs(val_targets - val_predictions))
            all_val_metrics.append({'rmse': fold_rmse, 'mae': fold_mae})
            
            print(f"Fold {fold} Metrics:")
            print(f"RMSE: ${fold_rmse:.2f}")
            print(f"MAE: ${fold_mae:.2f}")
        
        # Calculate and plot average training history
        avg_train_losses = np.mean(all_train_losses, axis=0)
        avg_val_losses = np.mean(all_val_losses, axis=0)
        
        plt.figure(figsize=(12, 5))
        plt.plot(avg_train_losses, label='Avg Training Loss')
        plt.plot(avg_val_losses, label='Avg Validation Loss')
        plt.fill_between(range(len(avg_train_losses)),
                        np.min(all_train_losses, axis=0),
                        np.max(all_train_losses, axis=0),
                        alpha=0.2)
        plt.fill_between(range(len(avg_val_losses)),
                        np.min(all_val_losses, axis=0),
                        np.max(all_val_losses, axis=0),
                        alpha=0.2)
        plt.title('Average Model Loss Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('plots/moirai_cv_training_history.png')
        plt.close()
        
        # Print average metrics across folds
        avg_rmse = np.mean([m['rmse'] for m in all_val_metrics])
        avg_mae = np.mean([m['mae'] for m in all_val_metrics])
        rmse_std = np.std([m['rmse'] for m in all_val_metrics])
        mae_std = np.std([m['mae'] for m in all_val_metrics])
        
        print("\nCross-Validation Results:")
        print(f"Average RMSE: ${avg_rmse:.2f} ± ${rmse_std:.2f}")
        print(f"Average MAE: ${avg_mae:.2f} ± ${mae_std:.2f}")
        
        return all_train_losses, all_val_losses, all_val_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        dataset = TimeSeriesDataset(X, self.sequence_length)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                predictions.append(output[:, -1, 0].cpu().numpy())
        
        predictions = np.concatenate(predictions)
        
        # Create dummy array for inverse transform
        dummy_array = np.zeros((len(predictions), X.shape[1]))
        dummy_array[:, 0] = predictions  # Close price is the first column
        predictions_transformed = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        return predictions_transformed
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate various performance metrics"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        # Calculate directional accuracy
        y_true_dir = np.sign(np.diff(y_true))
        y_pred_dir = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(y_true_dir == y_pred_dir) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot predictions vs actual values"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        # Plot predictions
        ax1.plot(y_true, label='Actual', color='blue')
        ax1.plot(y_pred, label='Predicted', color='red', linestyle='--')
        ax1.set_title('Stock Price Predictions (Moirai Transformer)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot prediction error
        error = y_pred - y_true
        ax2.plot(error, color='green', label='Prediction Error')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.fill_between(range(len(error)), error, 0, 
                        where=(error >= 0), color='green', alpha=0.3)
        ax2.fill_between(range(len(error)), error, 0,
                        where=(error < 0), color='red', alpha=0.3)
        ax2.set_title('Prediction Error')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Error')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/moirai_predictions.png')
        plt.close()

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    
    # Load the processed data
    data = pd.read_csv('processed_data/AAPL_processed.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Calculate additional technical indicators
    print("Calculating technical indicators...")
    
    # EMA and MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    # Momentum Indicators
    data['MOM'] = data['Close'].diff(10)
    data['ROC'] = data['Close'].pct_change(10) * 100
    
    # Volume-based Indicators
    data['OBV'] = (data['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * data['Volume']).cumsum()
    data['VWAP'] = (data['Close'] * data['Volume']).rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()
    
    # Price derivatives
    data['Price_Change'] = data['Close'].pct_change()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    
    # Drop rows with NaN values
    data = data.dropna()
    
    print(f"Final dataset shape: {data.shape}")
    
    # Prepare data and train model
    print("\nPreparing data and training model...")
    predictor = MoiraiPredictor(
        sequence_length=20,
        batch_size=32,
        learning_rate=0.0005,
        num_epochs=200
    )
    X = predictor.prepare_data(data)
    train_losses, val_losses, val_metrics = predictor.train(X, early_stopping_patience=15)
    
    # Make predictions on the entire dataset
    print("\nMaking predictions...")
    predictions = predictor.predict(X)
    
    # Calculate metrics
    metrics = predictor.calculate_metrics(data['Close'].values[predictor.sequence_length:], predictions)
    
    print("\nModel Performance Metrics:")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"MAE: ${metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    
    # Plot predictions
    print("\nGenerating prediction plots...")
    predictor.plot_predictions(
        data['Close'].values[predictor.sequence_length:],
        predictions
    )
    
    print("\nAnalysis complete! Check 'moirai_predictions.png' for visualization.")
    print("Training history has been saved as 'moirai_cv_training_history.png'")
    print("Prediction plot has been saved as 'moirai_predictions.png'")
    
    # Calculate and plot the prediction error distribution
    errors = data['Close'].values[predictor.sequence_length:] - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error ($)')
    plt.ylabel('Density')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.savefig('plots/moirai_error_distribution.png')
    plt.close()
    
    print("Error distribution plot has been saved as 'moirai_error_distribution.png'")
