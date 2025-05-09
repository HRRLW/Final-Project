import os
import pandas as pd
import numpy as np
from fredapi import Fred
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

class DataPreprocessor:
    def __init__(self, fred_api_key='fb191aed4848289422bc99dfb6710e0f'):
        self.fred = Fred(api_key=fred_api_key)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def load_stock_data(self, symbol, start_date='2019-01-01', end_date='2022-12-31'):
        """Load stock data from local CSV file"""
        file_path = f'raw_data/{symbol}.csv'
        df = pd.read_csv(file_path)
        
        # Convert date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        return df
    
    def load_economic_indicators(self, start_date='2019-01-01', end_date='2022-12-31'):
        """Load economic indicators from FRED"""
        indicators = {
            'UNRATE': 'Unemployment_Rate',
            'CPIAUCSL': 'CPI',
            'FEDFUNDS': 'Fed_Funds_Rate',
            'SP500': 'SP500'
        }
        
        economic_data = pd.DataFrame()
        for series_id, name in indicators.items():
            series = self.fred.get_series(series_id, start_date, end_date)
            economic_data[name] = series
            
        # Forward fill missing values (as economic data is not daily)
        economic_data = economic_data.resample('D').ffill()
        return economic_data
    
    def process_news_sentiment(self, headlines_data):
        """Process news headlines and calculate sentiment scores"""
        sentiments = []
        for headline in headlines_data:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(headline)
            sentiments.append(sentiment_scores['compound'])
        return np.mean(sentiments)
    
    def prepare_features(self, stock_symbol, start_date='2019-01-01', end_date='2022-12-31'):
        """Prepare final dataset with all features"""
        # Load stock data
        stock_data = self.load_stock_data(stock_symbol, start_date, end_date)
        
        # Calculate technical indicators
        stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['RSI'] = self.calculate_rsi(stock_data['Close'])
        
        # Load economic indicators
        economic_data = self.load_economic_indicators(start_date, end_date)
        
        # Merge all data
        final_data = pd.merge(stock_data, economic_data, 
                            left_on='Date', right_index=True, how='left')
        
        # Drop rows with missing values
        final_data = final_data.dropna()
        
        return final_data
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Prepare data for a specific stock (e.g., AAPL)
    data = preprocessor.prepare_features('AAPL')
    print("Sample of prepared data:")
    print(data.head())
    print("\nFeatures available:", data.columns.tolist())
    
    # Save the processed data
    os.makedirs('processed_data', exist_ok=True)
    data.to_csv('processed_data/AAPL_processed.csv', index=False)
