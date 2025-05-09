import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

class AnalysisUtils:
    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing various analysis results
        """
        # Basic statistics
        stats_dict = {
            'time_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d'),
                'trading_days': len(df)
            },
            'missing_values': df.isnull().sum().to_dict(),
            'basic_stats': {}
        }
        
        # Convert describe() output to JSON serializable format
        numeric_df = df.select_dtypes(include=[np.number])
        desc = numeric_df.describe()
        for col in desc.columns:
            stats_dict['basic_stats'][col] = {
                'count': int(desc[col]['count']),
                'mean': float(desc[col]['mean']),
                'std': float(desc[col]['std']),
                'min': float(desc[col]['min']),
                '25%': float(desc[col]['25%']),
                '50%': float(desc[col]['50%']),
                '75%': float(desc[col]['75%']),
                'max': float(desc[col]['max'])
            }
        
        # Calculate market conditions
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['Market_Condition'] = 'Normal'
        df.loc[df['Volatility'] > df['Volatility'].quantile(0.75), 'Market_Condition'] = 'High_Volatility'
        df.loc[df['Volatility'] < df['Volatility'].quantile(0.25), 'Market_Condition'] = 'Low_Volatility'
        
        # Market condition statistics
        stats_dict['market_conditions'] = df['Market_Condition'].value_counts().to_dict()
        
        # Save analysis results
        os.makedirs('results', exist_ok=True)  # Ensure results directory exists
        with open('results/analysis_results.json', 'w') as f:
            json.dump(stats_dict, f, indent=4)
        
        return stats_dict
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame) -> None:
        """
        Plot correlation heatmap for features
        
        Args:
            df: Input DataFrame
        """
        plt.figure(figsize=(12, 10))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('plots/correlation_heatmap.png')
        plt.close()
    
    @staticmethod
    def plot_feature_importance(model_name: str, importance_data: pd.DataFrame) -> None:
        """
        Plot feature importance
        
        Args:
            model_name: Name of the model
            importance_data: DataFrame with feature importance data
        """
        os.makedirs('plots', exist_ok=True)  # Ensure plots directory exists
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_data, x='importance', y='feature')
        plt.title(f'{model_name} Feature Importance')
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.lower()}_feature_importance.png')
        plt.close()
    
    @staticmethod
    def calculate_confidence_intervals(predictions: np.ndarray, 
                                    actual: np.ndarray, 
                                    confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence intervals for model metrics
        
        Args:
            predictions: Model predictions
            actual: Actual values
            confidence: Confidence level
            
        Returns:
            Dictionary containing confidence intervals
        """
        errors = actual - predictions
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin_of_error = z_score * (std_error / np.sqrt(len(errors)))
        
        return {
            'mean_error': mean_error,
            'confidence_interval': (mean_error - margin_of_error, mean_error + margin_of_error),
            'confidence_level': confidence
        }
    
    @staticmethod
    def evaluate_market_conditions(predictions: np.ndarray, 
                                 actual: np.ndarray, 
                                 market_conditions: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance under different market conditions
        
        Args:
            predictions: Model predictions
            actual: Actual values
            market_conditions: Array of market conditions
            
        Returns:
            Dictionary containing performance metrics for each market condition
        """
        conditions = np.unique(market_conditions)
        results = {}
        
        for condition in conditions:
            mask = market_conditions == condition
            if np.sum(mask) > 0:
                results[condition] = {
                    'rmse': np.sqrt(mean_squared_error(actual[mask], predictions[mask])),
                    'mae': mean_absolute_error(actual[mask], predictions[mask]),
                    'r2': r2_score(actual[mask], predictions[mask]),
                    'samples': np.sum(mask)
                }
        
        return results
    
    @staticmethod
    def plot_market_condition_performance(market_metrics: Dict[str, Dict[str, float]], 
                                        model_name: str) -> None:
        """
        Plot model performance under different market conditions
        
        Args:
            market_metrics: Dictionary containing performance metrics for each market condition
            model_name: Name of the model
        """
        conditions = list(market_metrics.keys())
        metrics = ['rmse', 'mae', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [market_metrics[c][metric] for c in conditions]
            axes[i].bar(conditions, values)
            axes[i].set_title(f'{metric.upper()} by Market Condition')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'{model_name} Performance Across Market Conditions')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_market_performance.png')
        plt.close()
    
    @staticmethod
    def save_model_architecture(model: torch.nn.Module, filename: str) -> None:
        """
        Save model architecture visualization
        
        Args:
            model: PyTorch model
            filename: Output filename
        """
        try:
            from torchviz import make_dot
            
            # Create dummy input
            batch_size = 1
            seq_length = 10
            n_features = model.input_dim if hasattr(model, 'input_dim') else 14
            
            x = torch.randn(batch_size, seq_length, n_features)
            y = model(x)
            
            # Generate visualization
            dot = make_dot(y, params=dict(model.named_parameters()))
            dot.render(filename, format='png', cleanup=True)
            
        except ImportError:
            print("Please install torchviz package to visualize model architecture")
    
    @staticmethod
    def plot_training_progress(train_losses: List[float], 
                             val_losses: List[float], 
                             train_metrics: List[float], 
                             val_metrics: List[float],
                             metric_name: str,
                             model_name: str) -> None:
        """
        Plot training progress with multiple metrics
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            train_metrics: Training metrics
            val_metrics: Validation metrics
            metric_name: Name of the metric
            model_name: Name of the model
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot metrics
        ax2.plot(train_metrics, label=f'Training {metric_name}')
        ax2.plot(val_metrics, label=f'Validation {metric_name}')
        ax2.set_title(f'Model {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        
        plt.suptitle(f'{model_name} Training Progress')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_training_progress.png')
        plt.close()
