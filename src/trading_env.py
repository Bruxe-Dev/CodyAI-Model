import numpy as np 
import yfinance as yf 
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from enum import Enum

class TradingAction(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class TradingEnv:
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
        
        close_prices = df['close'].squeeze()
        rsi = RSIIndicator(close=close_prices, window=14)
        df['RSI'] = rsi.rsi()
        
        macd = MACD(close=close_prices)
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

    def get_state(self):
                row = self.stock_data.iloc[self.current_step]
        
            # Normalize price to 0-1 range using min-max over recent window
                recent_prices = self.stock_data['Close'].iloc[max(0, self.current_step-50):self.current_step+1]
                price_min = recent_prices.min()
                price_max = recent_prices.max()
                normalized_price = (row['Close'] - price_min) / (price_max - price_min + 1e-8)
                
                # Normalize volume
                recent_volume = self.stock_data['Volume'].iloc[max(0, self.current_step-50):self.current_step+1]
                volume_min = recent_volume.min()
                volume_max = recent_volume.max()
                normalized_volume = (row['Volume'] - volume_min) / (volume_max - volume_min + 1e-8)
                
                # Portfolio state
                current_price = row['Close']
                portfolio_value = self.balance + (self.shares_held * current_price)
                portfolio_ratio = portfolio_value / self.initial_balance  # How much profit/loss
                
                # Position ratio (how much stock vs cash)
                stock_value = self.shares_held * current_price
                position_ratio = stock_value / (portfolio_value + 1e-8)
                
                # Days holding position
                # (We'll track this later - for now just use 0)
                days_held = 0
                
                state = [
                    # Market indicators (8 features)
                    normalized_price,           # 0: Current price (normalized)
                    normalized_volume,          # 1: Volume (normalized)
                    row['RSI'] / 100,          # 2: RSI (0-1 range)
                    np.tanh(row['MACD'] / 100), # 3: MACD (normalized with tanh)
                    np.tanh(row['MACD_Signal'] / 100), # 4: MACD Signal
                    np.tanh(row['MACD_Diff'] / 100),   # 5: MACD Difference
                    (row['Close'] - row['SMA_20']) / (row['SMA_20'] + 1e-8), # 6: Distance from SMA_20
                    (row['Close'] - row['SMA_50']) / (row['SMA_50'] + 1e-8), # 7: Distance from SMA_50
                    
                    # Portfolio state (4 features)
                    portfolio_ratio,            # 8: Total value vs initial (profit/loss)
                    position_ratio,             # 9: How much in stocks vs cash
                    float(self.shares_held > 0), # 10: Do we own stock? (1 or 0)
                    days_held / 100,            # 11: How long we've held (normalized)
                    
                    # Recent momentum (3 features)
                    row['Price_Change'],        # 12: Today's price change %
                    row['Volume_Change'],       # 13: Today's volume change %
                    np.tanh((row['Close'] - self.stock_data['Close'].iloc[max(0, self.current_step-5):self.current_step].mean()) / row['Close']) # 14: 5-day momentum
                ]
                
                return np.array(state, dtype=np.float32)

    def step(self, action):

            current_price = self.stock_data.iloc[self.current_step]['Close']
            
            reward = 0
            
            # Execute action
            if action == TradingAction.BUY.value:
                # BUY: Use all cash to buy shares
                if self.balance > 0 and self.shares_held == 0:
                    # Calculate how many shares we can buy
                    shares_to_buy = self.balance / (current_price * (1 + self.commission))
                    cost = shares_to_buy * current_price * (1 + self.commission)
                    
                    self.shares_held = shares_to_buy
                    self.balance -= cost
                    self.total_trades += 1
                    
                    # Small penalty for trading (to prevent overtrading)
                    reward = -0.1
            
            elif action == TradingAction.SELL.value:
                # SELL: Sell all shares
                if self.shares_held > 0:
                    # Calculate profit/loss
                    sale_value = self.shares_held * current_price * (1 - self.commission)
                    profit = sale_value - (self.initial_balance - self.balance)
                    
                    self.balance += sale_value
                    self.shares_held = 0
                    self.total_trades += 1
                    
                    # Reward based on profit (normalized)
                    reward = profit / self.initial_balance * 100  # Convert to percentage
            
            # HOLD: Do nothing (action == 0)
            
            # Move to next day
            self.current_step += 1
            
            # Calculate current portfolio value
            portfolio_value = self.balance + (self.shares_held * self.stock_data.iloc[self.current_step]['Close'])
            self.net_worth_history.append(portfolio_value)
            
            # Small reward for holding profitable position
            if self.shares_held > 0:
                unrealized_profit = (portfolio_value - self.initial_balance) / self.initial_balance
                reward += unrealized_profit * 0.1  # Small incremental reward
            
            # Check if episode is done
            done = self.current_step >= self.max_steps
            
            # Get next state
            next_state = self._get_state()
            
            return next_state, reward, done
    
    def get_portfolio_value(self):
        """Get current total portfolio value"""
        current_price = self.stock_data.iloc[self.current_step]['Close']
        return self.balance + (self.shares_held * current_price)
    
    def get_return(self):
        """Get total return percentage"""
        return ((self.get_portfolio_value() - self.initial_balance) / self.initial_balance) * 100


def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    
    print(f"✓ Downloaded {len(data)} days of data")
    
    return data


# Test the environment
if __name__ == "__main__":
    # Download Apple stock data
    stock_data = download_stock_data('AAPL', '2020-01-01', '2024-12-31')
    
    # Create trading environment
    env = TradingEnv(stock_data, initial_balance=10000)
    
    # Test one episode with random actions
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Initial state: {state}")
    print(f"\nStarting balance: ${env.initial_balance}")
    
    total_reward = 0
    
    for step in range(100):
        # Random action
        action = np.random.randint(0, 3)
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Action: {TradingAction(action).name}")
            print(f"  Portfolio Value: ${env.get_portfolio_value():.2f}")
            print(f"  Return: {env.get_return():.2f}%")
            print(f"  Shares Held: {env.shares_held:.2f}")
            print(f"  Cash: ${env.balance:.2f}")
        
        if done:
            break
    
    print(f"\n✓ Episode complete!")
    print(f"Final portfolio value: ${env.get_portfolio_value():.2f}")
    print(f"Total return: {env.get_return():.2f}%")
    print(f"Total trades: {env.total_trades}")

    