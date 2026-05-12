import MetaTrader5 as mt5
from datetime import datetime, time as dtime
import time
import pandas as pd
import numpy as np
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

from src_trade.trading_agent import TradingDQNAgent
from src_trade.trading_env import ForexEnv 
from src_trade import config

class OverlapSniperBot:
    def __init__(self):
        if not mt5.initialize():
            print("❌ MT5 Init Failed")
            quit()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.target_pairs = ["GBPUSD", "EURUSD", "USDJPY"]
        self.risk_pct = 0.01 
        self.trade_taken_today = False
        
        self.agent = TradingDQNAgent(input_size=31) 
        self.agent.load(config.MODEL_PATH)
        self.agent.epsilon = 0.0 # Strict Sniper Mode

    def get_gemini_market_outlook(self):
        try:
            print("🔍 Gemini is doing Sentimental Analysis...")
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents="Analyze the last 4 hours of economic news for USD, EUR, and GBP. "
                         "Return only a single number between -1.0 (Bearish) and 1.0 (Bullish)."
            )
            sentiment = float(response.text.strip())
            print(f"📊 GEMINI SENTIMENT: {sentiment}") # Added 'f' for formatting
            return sentiment
        except Exception as e:
            print(f"⚠️ Gemini API Error: {e}")
            return 0.0

    def get_live_state(self, symbol, sentiment):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'TickVol', 'Spread', 'RealVol']
        
        temp_env = ForexEnv(df) 
        state = temp_env._get_state()
        
        state[30] = sentiment 
        return state

    def scan_for_volatility(self):
        best_pair = None
        max_vol = -1
        
        for symbol in self.target_pairs:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
            if rates is None: continue
            df = pd.DataFrame(rates)
        
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                 np.maximum(abs(df['high'] - df['close'].shift()), 
                                            abs(df['low'] - df['close'].shift())))
            atr = df['tr'].rolling(14).mean().iloc[-1]
            avg_range = df['tr'].rolling(20).mean().iloc[-1]
            
            rel_vol = atr / avg_range
            if rel_vol > max_vol:
                max_vol = rel_vol
                best_pair = symbol
        return best_pair

    def calculate_position_size(self, symbol, stop_loss_pips):
        """Calculates lot size based on 1% account risk"""
        account = mt5.account_info()
        risk_amount = account.balance * self.risk_pct
        
        # Standard Pip Value for majors is ~$10 for 1.0 lot
        # (Adjustment needed for USDJPY or non-USD accounts)
        pip_value = 10.0 
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        return round(max(0.01, min(lot_size, 10.0)), 2)

    def execute_one_shot(self, symbol, direction, sentiment):
        """Handles the single daily trade execution with ATR-based SL"""
        # 1. Get ATR for SL calculation
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 20)
        df = pd.DataFrame(rates)
        atr = (df['high'] - df['low']).mean()
        
        price = mt5.symbol_info_tick(symbol).ask if direction == "BUY" else mt5.symbol_info_tick(symbol).bid
        sl_dist = atr * 2.0 # 2x ATR for Stop Loss
        sl = price - sl_dist if direction == "BUY" else price + sl_dist
        tp = price + (sl_dist * 1.5) if direction == "BUY" else price - (sl_dist * 1.5)
        
        # Convert SL distance to pips for lot sizing (assuming 5-digit broker)
        sl_pips = sl_dist * 10000 
        lots = self.calculate_position_size(symbol, sl_pips)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 123456,
            "comment": f"Overlap Sniper | Sent: {sentiment}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ One-Shot Executed: {direction} {symbol} at {price}")
            self.trade_taken_today = True
            self.log_trade(symbol, direction, price, sentiment)
        else:
            print(f"❌ Trade Failed: {result.comment}")

    def log_trade(self, symbol, direction, price, sentiment):
        log_data = {
            "Timestamp": datetime.now(),
            "Symbol": symbol,
            "Type": direction,
            "Price": price,
            "Sentiment": sentiment,
            "Status": "OPEN"
        }
        pd.DataFrame([log_data]).to_csv("professional_trading_log.csv", mode='a', header=False, index=False)

    def run(self):
        print("🤖 Sniper Bot Live. Waiting for 13:00 UTC...")
        while True:
            now_utc = datetime.utcnow().time()
            
            if dtime(13, 0) <= now_utc <= dtime(17, 0) and not self.trade_taken_today:
                # 2. SELECT BEST PAIR (Volatility Scanner)
                best_pair = self.scan_for_volatility()
                sentiment = self.get_gemini_market_outlook()
                
                # 3. GET LIVE STATE FOR THE SELECTED PAIR
                state = self.get_live_state(best_pair, sentiment)
                
                # 4. MANUALLY ALIGN OVERLAP FLAG (Index 29)
                state[29] = 1.0 
                state[30] = sentiment 
                
                action = self.agent.act(state)
      
                if action == 1 and sentiment > 0.1:
                    print(f"🚀 EXECUTE BUY ON {best_pair}")
                    self.execute_one_shot(best_pair, "BUY", sentiment)
                elif action == 2 and sentiment < -0.1:
                    print(f"📉 EXECUTE SELL ON {best_pair}")
                    self.execute_one_shot(best_pair, "SELL", sentiment)
                else:
                    print(f"📡 Waiting for Sniper Signal on {best_pair}...")
                
                time.sleep(300) # Check every 5 mins
            
            # 6. RESET AT END OF DAY
            if now_utc > dtime(17, 0):
                if self.trade_taken_today:
                    print("💤 Session ended. Trade was taken. Resetting for tomorrow.")
                self.trade_taken_today = False 
            
            time.sleep(60)
if __name__ == "__main__":
    bot = OverlapSniperBot()
    bot.run()