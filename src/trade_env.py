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
        