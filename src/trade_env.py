import numpy as np 
import yfinance as yf 
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from enum import Enum

class tradeAction(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

