"""
live_bot_pro.py  —  Overlap Sniper Bot

Sessions (UTC)
──────────────
PRIMARY   13:00–17:00   London/NY overlap   full risk   |sentiment| ≥ 0.10
SECONDARY 07:00–12:00   London open         50 % risk   |sentiment| ≥ 0.20
TERTIARY  17:00–22:00   NY late / Asia      25 % risk   |sentiment| ≥ 0.35
DEAD ZONE 22:00–07:00   no new entries

SL and TP are computed from market structure (swing highs/lows + ATR buffer)
via the same _compute_sl_tp() logic used in training — so live behaviour
exactly matches what the model learned.
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
from src_trade.trading_env_fixed   import ForexEnv
from src_trade import config


SESSIONS = {
    "PRIMARY":   {"start": dtime(13,0), "end": dtime(17,0),
                  "risk_mult": 1.00, "sent_thresh": 0.10, "is_primary": True},
    "SECONDARY": {"start": dtime( 7,0), "end": dtime(12,0),
                  "risk_mult": 0.50, "sent_thresh": 0.20, "is_primary": False},
    "TERTIARY":  {"start": dtime(17,0), "end": dtime(22,0),
                  "risk_mult": 0.25, "sent_thresh": 0.35, "is_primary": False},
}


class OverlapSniperBot:

    def __init__(self):
        if not mt5.initialize():
            print("❌ MT5 init failed"); quit()

        self.client       = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.target_pairs = ["GBPUSD", "EURUSD", "USDJPY"]
        self.base_risk    = 0.01
        self.max_daily    = 3

        self.trades_today      = 0
        self._last_sent_hour   = -1
        self._cached_sentiment = 0.0
        self._last_scan_hour   = -1

        # 46 features now
        self.agent = TradingDQNAgent(input_size=46)
        self.agent.load(config.MODEL_PATH)
        self.agent.epsilon = 0.0

    # ── session ───────────────────────────────────────────────────────────────
    def _active_session(self, now: dtime):
        for name, s in SESSIONS.items():
            if s["start"] <= now < s["end"]:
                return name, s
        return None, None

    # ── sentiment ─────────────────────────────────────────────────────────────
    def _get_sentiment(self) -> float:
        hour = datetime.utcnow().hour
        if hour == self._last_sent_hour:
            return self._cached_sentiment
        try:
            print("🔍 Refreshing Gemini sentiment…")
            resp = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=(
                    "Analyse the last 4 hours of economic news for USD, EUR, and GBP. "
                    "Reply with ONE float between -1.0 (Bearish) and 1.0 (Bullish). "
                    "No explanation — just the number."
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

    # ── volatility scanner ────────────────────────────────────────────────────
    def _best_pair(self) -> 'str | None':
        best, top = None, -1.0
        for sym in self.target_pairs:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 50)
            if rates is None: continue
            df = pd.DataFrame(rates)
            df['tr'] = np.maximum(df['high']-df['low'],
                        np.maximum(abs(df['high']-df['close'].shift()),
                                   abs(df['low'] -df['close'].shift())))
            rel = df['tr'].rolling(14).mean().iloc[-1] / \
                  (df['tr'].rolling(20).mean().iloc[-1] + 1e-8)
            if rel > top:
                top, best = rel, sym
        return best

    # ── build live state (46 features) ───────────────────────────────────────
    def _live_state(self, symbol: str, sentiment: float,
                    is_primary: bool) -> np.ndarray:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 300)
        df    = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.columns = ['Open','High','Low','Close','TickVol','Spread','RealVol']
        tmp         = ForexEnv(df)
        state       = tmp._get_state()
        state[44]   = 1.0 if is_primary else 0.0
        state[45]   = float(sentiment)
        return state

    # ── compute SL/TP using same logic as training env ─────────────────────────
    def _structure_sl_tp(self, symbol: str, direction: int,
                         entry_price: float) -> 'tuple[float,float,float]':
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 120)
        df    = pd.DataFrame(rates)
        hi, lo, cl = df['high'], df['low'], df['close']
        atr   = float((hi - lo).rolling(14).mean().iloc[-1])
        sh50  = float(hi.rolling(50).max().iloc[-1])
        sl50  = float(lo.rolling(50).min().iloc[-1])
        sh100 = float(hi.rolling(100).max().iloc[-1])
        sl100 = float(lo.rolling(100).min().iloc[-1])

        if direction == 1:
            sl_p    = sl50  - atr * 0.5
            tp_p    = sh50  - atr * 0.2
            sl_dist = entry_price - sl_p
            tp_dist = tp_p - entry_price
            if sl_dist <= 0 or tp_dist / (sl_dist+1e-8) < 1.5:
                tp_p    = sh100 - atr * 0.2
                tp_dist = tp_p - entry_price
        else:
            sl_p    = sh50  + atr * 0.5
            tp_p    = sl50  + atr * 0.2
            sl_dist = sl_p - entry_price
            tp_dist = entry_price - tp_p
            if sl_dist <= 0 or tp_dist / (sl_dist+1e-8) < 1.5:
                tp_p    = sl100 + atr * 0.2
                tp_dist = entry_price - tp_p

        rr = tp_dist / (sl_dist + 1e-8)
        return sl_p, tp_p, rr

    # ── position sizing ───────────────────────────────────────────────────────
    def _lot_size(self, symbol: str, sl_price: float,
                  entry_price: float, risk_mult: float) -> float:
        acct     = mt5.account_info()
        risk_amt = acct.balance * self.base_risk * risk_mult
        sl_dist  = abs(entry_price - sl_price)
        # pip value ≈ $10/lot for 4-digit pairs; adjust for JPY
        pip_val  = 10.0 if 'JPY' not in symbol else 0.091
        pip_size = 0.0001 if 'JPY' not in symbol else 0.01
        sl_pips  = sl_dist / pip_size
        lot      = risk_amt / (sl_pips * pip_val + 1e-8)
        return round(max(0.01, min(lot, 5.0)), 2)

    # ── execute ───────────────────────────────────────────────────────────────
    def _execute(self, symbol: str, direction: str,
                 sentiment: float, risk_mult: float):
        tick      = mt5.symbol_info_tick(symbol)
        entry     = tick.ask if direction == "BUY" else tick.bid
        dir_int   = 1 if direction == "BUY" else -1
        sl, tp, rr = self._structure_sl_tp(symbol, dir_int, entry)

        if rr < 1.5:
            print(f"  ⚠️  RR={rr:.2f} < 1.5 — skipping (bad structure).")
            return

        lots = self._lot_size(symbol, sl, entry, risk_mult)
        print(f"  📐 Structure SL={sl:.5f}  TP={tp:.5f}  RR={rr:.2f}  Lots={lots}")

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lots,
            "type":         mt5.ORDER_TYPE_BUY if direction=="BUY" else mt5.ORDER_TYPE_SELL,
            "price":        entry,
            "sl":           round(sl, 5),
            "tp":           round(tp, 5),
            "magic":        123456,
            "comment":      f"Sniper|{direction}|RR{rr:.1f}|S{sentiment:.2f}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ {direction} {symbol} @ {entry:.5f}  SL={sl:.5f}  TP={tp:.5f}")
            self.trades_today += 1
            self._log(symbol, direction, entry, sl, tp, rr, sentiment, risk_mult)
        else:
            print(f"  ❌ Order failed ({result.retcode}): {result.comment}")

    def _log(self, symbol, direction, entry, sl, tp, rr, sentiment, risk_mult):
        pd.DataFrame([{
            "Timestamp": datetime.utcnow(), "Symbol": symbol,
            "Direction": direction, "Entry": entry,
            "SL": sl, "TP": tp, "RR": round(rr,2),
            "Sentiment": sentiment, "RiskMult": risk_mult,
            "DailyCount": self.trades_today,
        }]).to_csv("sniper_trade_log.csv", mode='a', header=False, index=False)

    # ── hourly scan ───────────────────────────────────────────────────────────
    def _hourly_scan(self, now: dtime):
        sess_name, sess = self._active_session(now)
        if sess_name is None:
            print(f"  💤 Dead zone — no scan."); return
        if self.trades_today >= self.max_daily:
            print(f"  🛑 Daily cap ({self.max_daily}) reached."); return

        risk_mult   = sess["risk_mult"]
        sent_thresh = sess["sent_thresh"]
        is_primary  = sess["is_primary"]

        print(f"\n🕐 [{now.strftime('%H:%M')} UTC] {sess_name}  "
              f"risk={risk_mult*100:.0f}%  |sent|≥{sent_thresh}")

        sentiment = self._get_sentiment()
        symbol    = self._best_pair()
        if symbol is None:
            print("  ⚠️  No valid pair."); return

        state  = self._live_state(symbol, sentiment, is_primary)
        action = self.agent.act(state)
        label  = ["HOLD","BUY","SELL"][action]
        print(f"  Pair: {symbol}  AI: {label}  Sentiment: {sentiment:+.3f}")

        if action == 1 and sentiment >  sent_thresh:
            self._execute(symbol, "BUY",  sentiment, risk_mult)
        elif action == 2 and sentiment < -sent_thresh:
            self._execute(symbol, "SELL", sentiment, risk_mult)
        else:
            print("  📡 No clean setup — skip.")

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print("🤖 Sniper Bot live — RR-based entries, structure SL/TP.")
        print(f"   Daily cap: {self.max_daily}  |  Dead zone: 22:00–07:00 UTC\n")
        while True:
            now   = datetime.utcnow()
            now_t = now.time()
            now_h = now.hour
            if now_t < dtime(0,2) and self.trades_today > 0:
                print("🔄 Midnight — resetting daily counter.")
                self.trades_today = 0
            if now.minute < 2 and now_h != self._last_scan_hour:
                self._last_scan_hour = now_h
                self._hourly_scan(now_t)
            time.sleep(30)


if __name__ == "__main__":
    bot = OverlapSniperBot()
    bot.run()