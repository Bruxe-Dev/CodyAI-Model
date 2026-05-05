import numpy as np 
import yfinance as yf 
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from enum import Enum

class tradingAction(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class tradingEnv:
    def __init__(self,stock_data, initial_balance = 10000,commission= 0.001):
        self.stock_data = stock_data
        self.initial_balance = initial_balance
        self.commission = commission

        self.calculate_indicators()

         # Tradng state
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_trades = 0
        self.net_worth_history = []
        
        # Episode tracking
        self.max_steps = len(self.stock_data) - 1

    def calculate_indicators(self):
        df = self.stock_data
        
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()
        
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # Simple Moving Averages - Average price over time
        df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20-day average
        df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day average
        
        # Price Change Percentage
        df['Price_Change'] = df['Close'].pct_change()
        
        # Volume Change
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Remove NaN values (first rows don't have enough data)
        df.dropna(inplace=True)
        
        self.stock_data = df
        self.max_steps = len(df) - 1

        def reset(self):

            self.current_step = 50  # Start at day 50 (need history for indicators)
            self.balance = self.initial_balance
            self.shares_held = 0
            self.total_trades = 0
            self.net_worth_history = [self.initial_balance]
            
            return self._get_state()