import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class MT5Interface:
    def __init__(self, symbol='EURUSD', timeframe='H1', account=None, password=None, server=None):

        self.symbol = symbol
        
        # Map timeframe strings to MT5 constants
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        self.timeframe = self.timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
        
        # Initialize MT5
        if not mt5.initialize():
            print(f" MT5 initialization failed: {mt5.last_error()}")
            return
        
        # Login if credentials provided
        if account and password and server:
            if not mt5.login(account, password, server):
                print(f" MT5 login failed: {mt5.last_error()}")
                return
        
        print(f"✅ MT5 connected successfully!")
        print(f"   Account: {mt5.account_info().login}")
        print(f"   Balance: ${mt5.account_info().balance}")
        print(f"   Server: {mt5.account_info().server}")
    
    def get_live_data(self, bars=500):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        
        if rates is None:
            print(f"❌ Failed to get data: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to match your env
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        })
        
        # Add candlestick pattern recognition
        df = self.detect_candlestick_patterns(df)
        
        return df
    
    def detect_candlestick_patterns(self, df):
        """
        Detect common candlestick patterns
        
        Patterns detected:
        - Doji
        - Hammer / Inverted Hammer
        - Engulfing (Bullish / Bearish)
        - Morning Star / Evening Star
        """
        # Calculate candle body and wicks
        df['Body'] = abs(df['Close'] - df['Open'])
        df['UpperWick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['LowerWick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Range'] = df['High'] - df['Low']
        
        # Doji: Small body relative to range
        df['Doji'] = (df['Body'] / df['Range'] < 0.1).astype(int)
        
        # Hammer: Long lower wick, small body at top
        df['Hammer'] = (
            (df['LowerWick'] > 2 * df['Body']) & 
            (df['UpperWick'] < df['Body'])
        ).astype(int)
        
        # Inverted Hammer: Long upper wick, small body at bottom
        df['InvertedHammer'] = (
            (df['UpperWick'] > 2 * df['Body']) & 
            (df['LowerWick'] < df['Body'])
        ).astype(int)
        
        # Bullish Engulfing: Current green candle engulfs previous red
        df['BullishEngulfing'] = 0
        for i in range(1, len(df)):
            if (df.iloc[i]['Close'] > df.iloc[i]['Open'] and  # Green candle
                df.iloc[i-1]['Close'] < df.iloc[i-1]['Open'] and  # Previous red
                df.iloc[i]['Close'] > df.iloc[i-1]['Open'] and  # Engulfs top
                df.iloc[i]['Open'] < df.iloc[i-1]['Close']):  # Engulfs bottom
                df.at[i, 'BullishEngulfing'] = 1
        
        # Bearish Engulfing: Current red candle engulfs previous green
        df['BearishEngulfing'] = 0
        for i in range(1, len(df)):
            if (df.iloc[i]['Close'] < df.iloc[i]['Open'] and  # Red candle
                df.iloc[i-1]['Close'] > df.iloc[i-1]['Open'] and  # Previous green
                df.iloc[i]['Close'] < df.iloc[i-1]['Open'] and  # Engulfs bottom
                df.iloc[i]['Open'] > df.iloc[i-1]['Close']):  # Engulfs top
                df.at[i, 'BearishEngulfing'] = 1
        
        return df
    
    def place_order(self, action, volume=0.01):
        """
        Place live order on MT5
        
        Args:
            action: 0=close, 1=buy, 2=sell
            volume: Lot size (0.01 = micro lot)
        
        Returns:
            Order result
        """
        # Get current position
        positions = mt5.positions_get(symbol=self.symbol)
        
        # CLOSE position
        if action == 0 and len(positions) > 0:
            for position in positions:
                # Determine close type (opposite of open)
                close_type = mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": position.volume,
                    "type": close_type,
                    "position": position.ticket,
                    "magic": 234000,
                    "comment": "AI Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                return result
        
        # BUY
        elif action == 1 and len(positions) == 0:
            price = mt5.symbol_info_tick(self.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "magic": 234000,
                "comment": "AI Buy",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            return result
        
        # SELL
        elif action == 2 and len(positions) == 0:
            price = mt5.symbol_info_tick(self.symbol).bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                "magic": 234000,
                "comment": "AI Sell",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            return result
    
    def get_account_info(self):
        """Get current account status"""
        info = mt5.account_info()
        return {
            'balance': info.balance,
            'equity': info.equity,
            'profit': info.profit,
            'margin': info.margin,
            'margin_free': info.margin_free,
        }
    
    def get_current_position(self):
        """Get current open position"""
        positions = mt5.positions_get(symbol=self.symbol)
        if len(positions) > 0:
            pos = positions[0]
            return {
                'type': 'BUY' if pos.type == 0 else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'profit': pos.profit,
                'swap': pos.swap,
            }
        return None
    
    def shutdown(self):
        """Close MT5 connection"""
        mt5.shutdown()
        print(" MT5 disconnected")


# Test connection
if __name__ == "__main__":
    # Initialize MT5 (will use current logged-in account)
    mt5_client = MT5Interface(symbol='EURUSD', timeframe='H1')
    
    # Get live data
    data = mt5_client.get_live_data(bars=100)
    
    if data is not None:
        print(f"\n Got {len(data)} candlesticks")
        print(f"\nLatest 5 candles:")
        print(data[['time', 'Open', 'High', 'Low', 'Close', 'Doji', 'Hammer']].tail())
        
        # Get account info
        account = mt5_client.get_account_info()
        print(f"\n📊 Account Info:")
        print(f"   Balance: ${account['balance']:.2f}")
        print(f"   Equity: ${account['equity']:.2f}")
        print(f"   Profit: ${account['profit']:.2f}")
    
    mt5_client.shutdown()