import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import MetaTrader5 as mt5

class ForexEnv:
    def __init__(self, forex_data, initial_balance=10000, leverage=3):
        self.forex_data = forex_data
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.position_size_pct = 0.2 
        self._calculate_indicators()
        self.reset()

    def _calculate_indicators(self):
        df = self.forex_data.copy()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['RSI_Norm'] = (df['RSI'] - 50) / 50
        macd = MACD(close=df['Close'])
        df['MACD_Norm'] = np.tanh(macd.macd_diff() / 0.01)
        
        for w in [12, 26, 50]:
            df[f'EMA_{w}'] = EMAIndicator(close=df['Close'], window=w).ema_indicator()
            df[f'EMA_{w}_Dist'] = (df['Close'] - df[f'EMA_{w}']) / (df[f'EMA_{w}'] + 1e-8)
        
        df['Swing_High'] = df['High'].rolling(window=50).max()
        df['Swing_Low'] = df['Low'].rolling(window=50).min()
        s_range = df['Swing_High'] - df['Swing_Low'] + 1e-8
        df['Fib_618_Buy'] = df['Swing_High'] - (s_range * 0.618)
        df['Dist_to_Fib618'] = (df['Close'] - df['Fib_618_Buy']) / (df['Close'] + 1e-8)
        df['Fib_618_Sell'] = df['Swing_Low'] + (s_range * 0.618)
        df['Dist_to_SellFib'] = (df['Close'] - df['Fib_618_Sell']) / (df['Close'] + 1e-8)
        df['Price_Change'] = df['Close'].pct_change().fillna(0)
        
        df.dropna(inplace=True)
        self.forex_data = df 
        self.max_steps = len(self.forex_data) - 1

    def reset(self):
        self.current_step = 50
        self.balance = self.initial_balance
        self.position = 0 
        self.entry_price = 0
        self.last_trade_step = 0
        self.total_trades = 0
        return self._get_state()

    def _get_state(self):
        row = self.forex_data.iloc[self.current_step]
        prev = self.forex_data.iloc[self.current_step - 1]
        p_prev = self.forex_data.iloc[self.current_step - 2]
        
        recent_prices = self.forex_data['Close'].iloc[max(0, self.current_step-50):self.current_step+1]
        norm_price = (row['Close'] - recent_prices.min()) / (recent_prices.max() - recent_prices.min() + 1e-8)
        
        # Base Features (0-22)
        state = [
            float(norm_price), float(row['EMA_12_Dist']), float(row['EMA_26_Dist']), float(row['EMA_50_Dist']),
            float((row['High'] - row['Low']) / (row['Close'] + 1e-8)),
            float(row['RSI_Norm']), float(row['MACD_Norm']), float(row['Price_Change']),
            float(self.position), 0.0, float((self.get_portfolio_value()/self.initial_balance)-1),
            float(self.total_trades/100), float(np.tanh(row['Close'] - recent_prices.mean())), 
            0.0, 0.0, float(self.leverage/10), float(self.position_size_pct), 
            float(self.current_step/self.max_steps),
            float((row['Close'] - row['Swing_High']) / row['Close']),
            float((row['Close'] - row['Swing_Low']) / row['Close']),
            float(row['Dist_to_Fib618']), float(row['Dist_to_SellFib']), 0.0
        ]

        # Candlestick Logic (23-28)
        body = abs(row['Close'] - row['Open'])
        state.extend([
            1.0 if (row['Low'] < min(row['Open'], row['Close']) and (min(row['Open'], row['Close']) - row['Low']) > 2*body) else 0.0, # Hammer
            1.0 if (row['High'] > max(row['Open'], row['Close']) and (row['High'] - max(row['Open'], row['Close'])) > 2*body) else 0.0, # Star
            0.0, 0.0, 0.0, # Placeholders for Star/Engulfing
            1.0 if (row['Close'] > prev['Open'] and row['Open'] < prev['Close']) else 0.0 # Engulfing
        ])

        # Index 29: London/NY Overlap (CRITICAL FOR TRAINING)
        current_hour = self.forex_data.index[self.current_step].hour
        state.append(1.0 if (13 <= current_hour <= 17) else 0.0)

        # Index 30: Sentiment Placeholder
        state.append(0.0) 
        
        return np.array(state, dtype=np.float32)
        
    def step(self, action):
        current_row = self.forex_data.iloc[self.current_step]
        current_price = current_row['Close']
        reward = 0
        
        # Execute trade
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.total_trades += 1
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
            self.total_trades += 1
        elif action == 0 and self.position != 0:
            # Close trade
            pnl = ((current_price - self.entry_price) / self.entry_price) * self.position
            self.balance += self.balance * self.position_size_pct * pnl * self.leverage
            reward = pnl * 10.0
            self.position = 0

        # Unrealized reward
        if self.position != 0:
            reward = ((current_price - self.entry_price) / self.entry_price) * 10.0 * self.position
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_state(), reward, done

    def get_portfolio_value(self):
        if self.position == 0: return self.balance
        current_price = self.forex_data.iloc[self.current_step]['Close']
        pnl = ((current_price - self.entry_price) / self.entry_price) * self.position * self.leverage
        return self.balance + (self.balance * self.position_size_pct * pnl)

    def get_return(self):
        return ((self.get_portfolio_value() - self.initial_balance) / self.initial_balance) * 100

def download_forex_data(pair='EURUSD', start_year=2019, end_year=2022):
    """Fetches historical H1 data from MT5 for the requested range."""
    if not mt5.initialize():
        print("❌ MT5 Initialize failed")
        return None
    
    # Ensure symbol matches MT5 naming convention
    symbol = pair.replace('=X', '')
    start_dt = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, 12, 31)
    
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        print(f"❌ No data found for {symbol} between {start_year}-{end_year}.")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']].copy()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    print(f"✅ Successfully downloaded {len(df)} candles from MT5.")
    return df