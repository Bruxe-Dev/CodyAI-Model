"""
live_bot_pro.py  —  Overlap Sniper Bot (fixed)

Sessions scanned (UTC)
──────────────────────
PRIMARY   13:00–17:00   London/NY overlap   full risk   sentiment ≥ |0.10|
SECONDARY 07:00–12:00   London open         50 % risk   sentiment ≥ |0.20|
TERTIARY  17:00–22:00   NY late / Asia      25 % risk   sentiment ≥ |0.35|
DEAD ZONE 22:00–07:00   no new entries

The bot wakes on every new H1 candle (checked at minute < 2) and runs the
full pipeline: volatility scan → Gemini sentiment (cached per hour) → AI
decision → execute if conditions are met.

Fixes vs original
-----------------
FIX-BOT-1  Off-session scanning.  Original bot slept completely outside
           13–17 UTC.  Now it scans every H1 candle across all three session
           windows with position-sized risk appropriate to each session.
FIX-BOT-2  Sentiment cached per hour to avoid burning Gemini API quota on
           every 5-minute check.  Re-fetched once per new hour automatically.
FIX-BOT-3  Daily trade cap (max_daily_trades = 3) prevents the bot from
           over-trading on volatile days when every session fires.
FIX-BOT-4  Midnight UTC reset clears the daily trade counter cleanly.
FIX-BOT-5  Gemini model corrected from "gemini-3-flash-preview" (doesn't exist)
           to "gemini-2.0-flash".
FIX-BOT-6  state[29] (session flag) is set correctly per session type rather
           than always forced to 1.0.
"""

import time
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

from src_trade.trading_agent_fixed import TradingDQNAgent
from src_trade.trading_env_fixed import ForexEnv
from src_trade import config


# ── session definitions ────────────────────────────────────────────────────
SESSIONS = {
    "PRIMARY":   {"start": dtime(13, 0), "end": dtime(17, 0),
                  "risk_mult": 1.00, "sent_thresh": 0.10, "is_primary": True},
    "SECONDARY": {"start": dtime( 7, 0), "end": dtime(12, 0),
                  "risk_mult": 0.50, "sent_thresh": 0.20, "is_primary": False},
    "TERTIARY":  {"start": dtime(17, 0), "end": dtime(22, 0),
                  "risk_mult": 0.25, "sent_thresh": 0.35, "is_primary": False},
}


class OverlapSniperBot:

    def __init__(self):
        if not mt5.initialize():
            print("❌ MT5 init failed")
            quit()

        self.client       = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.target_pairs = ["GBPUSD", "EURUSD", "USDJPY"]
        self.base_risk    = 0.01          # 1 % of balance at full risk
        self.max_daily    = 3             # FIX-BOT-3: hard daily trade cap

        self.trades_today      = 0
        self._last_sent_hour   = -1       # FIX-BOT-2: sentiment cache
        self._cached_sentiment = 0.0
        self._last_scan_hour   = -1       # one scan per H1 candle

        self.agent = TradingDQNAgent(input_size=31)
        self.agent.load(config.MODEL_PATH)
        self.agent.epsilon = 0.0          # pure exploitation in live mode

    # ── session detection ──────────────────────────────────────────────────
    def _active_session(self, now: dtime):
        for name, s in SESSIONS.items():
            if s["start"] <= now < s["end"]:
                return name, s
        return None, None

    # ── Gemini sentiment (cached per hour) ────────────────────────────────
    def _get_sentiment(self, force: bool = False) -> float:
        hour = datetime.utcnow().hour
        if not force and hour == self._last_sent_hour:
            return self._cached_sentiment
        try:
            print("🔍 Refreshing Gemini sentiment…")
            resp = self.client.models.generate_content(
                model="gemini-2.0-flash",      # FIX-BOT-5: valid model name
                contents=(
                    "Analyse the last 4 hours of economic news for USD, EUR, and GBP. "
                    "Reply with ONE float between -1.0 (Bearish) and 1.0 (Bullish). "
                    "No explanation, just the number."
                )
            )
            val = float(resp.text.strip())
            self._cached_sentiment = val
            self._last_sent_hour   = hour
            print(f"📊 Sentiment: {val:+.3f}")
            return val
        except Exception as exc:
            print(f"⚠️  Gemini error: {exc}")
            return self._cached_sentiment

    # ── volatility scanner ─────────────────────────────────────────────────
    def _best_pair(self) -> str | None:
        best, top = None, -1.0
        for sym in self.target_pairs:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 50)
            if rates is None:
                continue
            df = pd.DataFrame(rates)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(abs(df['high'] - df['close'].shift()),
                           abs(df['low']  - df['close'].shift()))
            )
            rel = df['tr'].rolling(14).mean().iloc[-1] / (df['tr'].rolling(20).mean().iloc[-1] + 1e-8)
            if rel > top:
                top, best = rel, sym
        return best

    # ── build live state ───────────────────────────────────────────────────
    def _live_state(self, symbol: str, sentiment: float,
                    is_primary: bool) -> np.ndarray:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        df    = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'TickVol', 'Spread', 'RealVol']
        tmp         = ForexEnv(df)
        state       = tmp._get_state()
        state[29]   = 1.0 if is_primary else 0.0   # FIX-BOT-6
        state[30]   = sentiment
        return state

    # ── position sizing ────────────────────────────────────────────────────
    def _lot_size(self, symbol: str, sl_pips: float,
                  risk_mult: float) -> float:
        acct      = mt5.account_info()
        risk_amt  = acct.balance * self.base_risk * risk_mult
        lot       = risk_amt / (sl_pips * 10.0 + 1e-8)
        return round(max(0.01, min(lot, 5.0)), 2)

    # ── execute trade ──────────────────────────────────────────────────────
    def _execute(self, symbol: str, direction: str,
                 sentiment: float, risk_mult: float):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 20)
        df    = pd.DataFrame(rates)
        atr   = float((df['high'] - df['low']).mean())

        tick  = mt5.symbol_info_tick(symbol)
        price = tick.ask if direction == "BUY" else tick.bid
        sl    = price - atr * 2 if direction == "BUY" else price + atr * 2
        tp    = price + atr * 3 if direction == "BUY" else price - atr * 3
        lots  = self._lot_size(symbol, atr * 2 * 10_000, risk_mult)

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lots,
            "type":         mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "magic":        123456,
            "comment":      f"Sniper|{direction}|S:{sentiment:.2f}|R:{risk_mult}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ {direction} {symbol} @ {price:.5f}  "
                  f"SL={sl:.5f}  TP={tp:.5f}  lots={lots}")
            self.trades_today += 1
            self._log(symbol, direction, price, sentiment, risk_mult)
        else:
            print(f"  ❌ Order failed ({result.retcode}): {result.comment}")

    def _log(self, symbol, direction, price, sentiment, risk_mult):
        pd.DataFrame([{
            "Timestamp": datetime.utcnow(), "Symbol": symbol,
            "Direction": direction, "Price": price,
            "Sentiment": sentiment, "RiskMult": risk_mult,
            "DailyCount": self.trades_today,
        }]).to_csv("sniper_trade_log.csv", mode='a', header=False, index=False)

    # ── hourly scan (FIX-BOT-1: runs in ALL active sessions) ──────────────
    def _hourly_scan(self, now: dtime):
        sess_name, sess = self._active_session(now)

        if sess_name is None:
            print(f"  💤 Dead zone — no scan.")
            return

        if self.trades_today >= self.max_daily:
            print(f"  🛑 Daily cap ({self.max_daily}) reached.")
            return

        risk_mult   = sess["risk_mult"]
        sent_thresh = sess["sent_thresh"]
        is_primary  = sess["is_primary"]

        print(f"\n🕐 [{now.strftime('%H:%M')} UTC] {sess_name} scan  "
              f"risk={risk_mult*100:.0f}%  |sent|≥{sent_thresh}")

        sentiment = self._get_sentiment()
        symbol    = self._best_pair()
        if symbol is None:
            print("  ⚠️  No valid pair."); return

        state  = self._live_state(symbol, sentiment, is_primary)
        action = self.agent.act(state)
        label  = ["HOLD", "BUY", "SELL"][action]
        print(f"  Pair: {symbol}  AI: {label}  Sentiment: {sentiment:+.3f}")

        if action == 1 and sentiment >  sent_thresh:
            self._execute(symbol, "BUY",  sentiment, risk_mult)
        elif action == 2 and sentiment < -sent_thresh:
            self._execute(symbol, "SELL", sentiment, risk_mult)
        else:
            print("  📡 No clean setup — skip.")

    # ── main loop ──────────────────────────────────────────────────────────
    def run(self):
        print("🤖 Sniper Bot live — scanning every H1 candle.")
        print("   Dead zone: 22:00–07:00 UTC  |  Daily cap:", self.max_daily)
        while True:
            now     = datetime.utcnow()
            now_t   = now.time()
            now_h   = now.hour

            # FIX-BOT-4: midnight reset
            if now_t < dtime(0, 2) and self.trades_today > 0:
                print("🔄 Midnight — resetting daily counter.")
                self.trades_today = 0

            # One scan per new H1 candle (minute 0 or 1)
            if now.minute < 2 and now_h != self._last_scan_hour:
                self._last_scan_hour = now_h
                self._hourly_scan(now_t)

            time.sleep(30)


if __name__ == "__main__":
    bot = OverlapSniperBot()
    bot.run()