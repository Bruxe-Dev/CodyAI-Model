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