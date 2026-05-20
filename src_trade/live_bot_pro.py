"""
live_bot_pro.py  —  Full Trader Live Bot

The model now controls both entry AND exit.
Every new M15 candle during the PRIMARY session:
  - If flat:  ask model → BUY / SELL / HOLD
  - If in trade: ask model → CLOSE / HOLD

The model uses the same 18-feature state it was trained on,
including session progress, Fib distances, and RSI divergence.
It will choose to CLOSE when it recognises exit conditions
(near resistance, momentum fading, session ending).

Hard SL/TP remain as safety nets ONLY.
The model is expected to exit before hitting them.
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

SL_ATR     = 2.0
TP_ATR     = 4.0
BASE_RISK  = 0.01
MAX_DAILY  = 3


class OverlapSniperBot:

    def __init__(self):
        if not mt5.initialize():
            print("❌ MT5 init failed"); quit()

        self.client        = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.target_pairs  = ["GBPUSD", "EURUSD", "USDJPY"]
        self.trades_today  = 0
        self._sent_hour    = -1
        self._sentiment    = 0.0
        self._last_h1      = -1
        self._last_m15     = -1
        self._h1_bias      = 0
        self._h1_pair      = None
        self._h1_sess_mult = 1.0
        self._open_ticket  = None   # MT5 ticket of current open position

        self.agent = TradingDQNAgent(input_size=18)
        self.agent.load(config.MODEL_PATH)
        self.agent.epsilon = 0.0
        print("✅ Full Trader Bot loaded. Model controls entry AND exit.\n")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _active_session(self, now):
        for name, s in SESSIONS.items():
            if s["start"] <= now < s["end"]:
                return name, s
        return None, None

    def _get_sentiment(self):
        h = datetime.utcnow().hour
        if h == self._sent_hour: return self._sentiment
        try:
            r = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Analyse last 4h news for USD, EUR, GBP. "
                         "Reply with ONE float -1.0 to 1.0. No explanation.")
            self._sentiment = float(r.text.strip())
            self._sent_hour = h
            print(f"  📊 Sentiment: {self._sentiment:+.3f}")
        except Exception as e:
            print(f"  ⚠️ Gemini: {e}")
        return self._sentiment

    def _best_pair(self):
        best, top = None, -1.0
        for sym in self.target_pairs:
            r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 50)
            if r is None: continue
            df = pd.DataFrame(r)
            df['tr'] = np.maximum(df['high']-df['low'],
                        np.maximum(abs(df['high']-df['close'].shift()),
                                   abs(df['low'] -df['close'].shift())))
            rel = df['tr'].rolling(14).mean().iloc[-1] / \
                  (df['tr'].rolling(20).mean().iloc[-1] + 1e-8)
            if rel > top: top, best = rel, sym
        return best

    def _build_state(self, symbol, sentiment, is_primary, now_utc):
        """Build the 18-feature state from live MT5 data."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 250)
        if rates is None: return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.columns = ['Open','High','Low','Close','TickVol','Spread','RealVol']

        # Build a temporary env just to compute indicators
        tmp = ForexEnv.__new__(ForexEnv)
        tmp.initial_balance   = 10_000
        tmp.leverage          = 10
        tmp.position_size_pct = 0.02
        tmp.position          = 0
        tmp.entry_price       = 0.0
        tmp.entry_sl          = 0.0
        tmp.entry_step        = 0
        tmp._step             = len(df) - 1
        tmp._max              = len(df)

        # Run indicators
        from ta.trend import EMAIndicator, ADXIndicator, MACD
        from ta.momentum import RSIIndicator
        from ta.volatility import AverageTrueRange
        c, h, l = df['Close'], df['High'], df['Low']
        df['ATR']   = AverageTrueRange(h,l,c,14).average_true_range()
        df['EMA20'] = EMAIndicator(c,20).ema_indicator()
        df['EMA50'] = EMAIndicator(c,50).ema_indicator()
        df['RSI']   = RSIIndicator(c,14).rsi()
        df['RSI3']  = RSIIndicator(c,3).rsi()
        df['MACD']  = MACD(c).macd_diff()
        df['ADX']   = ADXIndicator(h,l,c,14).adx()
        df['SH20']  = h.rolling(20).max()
        df['SL20']  = l.rolling(20).min()
        df['SH50']  = h.rolling(50).max()
        df['SL50']  = l.rolling(50).min()
        fib_rng     = df['SH50'] - df['SL50'] + 1e-8
        df['Fib618'] = df['SH50'] - fib_rng * 0.618
        df['Fib382'] = df['SH50'] - fib_rng * 0.382
        df.dropna(inplace=True)

        row = df.iloc[-1]
        atr = float(row['ATR']) + 1e-8
        cp  = float(row['Close'])

        # Get current position for unrealised PnL
        positions = mt5.positions_get(magic=123456) or []
        position  = 0; unreal_atr = 0.0; current_rr = 0.0; bars_held = 0.0
        if positions:
            pos = positions[0]
            position = 1 if pos.type == 0 else -1
            move     = (cp - pos.price_open) * position
            unreal_atr = move / atr
            sl_dist  = abs(pos.price_open - pos.sl) + 1e-8
            current_rr = move / sl_dist
            # Approximate bars held from time
            held_secs = (datetime.utcnow() - datetime.utcfromtimestamp(pos.time)).seconds
            bars_held = min(held_secs / 3600 / 24, 1.0)

        hour = now_utc.hour
        in_sess = 1.0 if 13 <= hour < 17 else 0.0
        sess_prog = (hour - 13) / 4 if in_sess else 0.0

        rsi_div = float(np.tanh((float(row['RSI3']) - float(row['RSI'])) / 20))

        state = np.array([
            float(np.tanh((cp - row['EMA50']) / (atr*3))),
            float(np.tanh((row['EMA20'] - row['EMA50']) / atr)),
            float((row['RSI'] - 50) / 50),
            float(np.tanh(row['MACD'] / atr)),
            float(row['ADX'] / 100),
            float(np.tanh(abs(cp - row['Open']) / atr)),
            float(1.0 if cp > row['Open'] else -1.0),
            float(np.tanh((row['SH20'] - cp) / atr)),
            float(np.tanh((cp - row['SL20']) / atr)),
            float(np.tanh((cp - row['Fib618']) / atr)),
            float(np.tanh((cp - row['Fib382']) / atr)),
            float(rsi_div),
            float(position),
            float(np.tanh(unreal_atr)),
            float(bars_held),
            float(np.tanh(current_rr)),
            float(sess_prog),
            float(in_sess),
        ], dtype=np.float32)

        return state, position, row, atr

    def _lot_size(self, symbol, sl_pips, risk_mult):
        acct = mt5.account_info()
        risk = acct.balance * BASE_RISK * risk_mult
        pip  = 0.01 if 'JPY' in symbol else 0.0001
        pv   = 0.091 if 'JPY' in symbol else 10.0
        lot  = risk / (sl_pips / pip * pv + 1e-8)
        return round(max(0.01, min(lot, 5.0)), 2)

    def _open_trade(self, symbol, direction, atr, risk_mult, sentiment):
        tick  = mt5.symbol_info_tick(symbol)
        entry = tick.ask if direction == "BUY" else tick.bid
        sl    = entry - atr*SL_ATR if direction=="BUY" else entry + atr*SL_ATR
        tp    = entry + atr*TP_ATR if direction=="BUY" else entry - atr*TP_ATR
        pip   = 0.0001
        lots  = self._lot_size(symbol, atr*SL_ATR, risk_mult)

        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if direction=="BUY" else mt5.ORDER_TYPE_SELL,
            "price": entry, "sl": round(sl,5), "tp": round(tp,5),
            "magic": 123456,
            "comment": f"AI|{direction}|S{sentiment:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ {direction} {symbol} @ {entry:.5f}  SL={sl:.5f}  TP={tp:.5f}")
            self.trades_today += 1
            self._open_ticket = result.order
            self._log(symbol, direction, entry, sl, tp, sentiment, risk_mult)
        else:
            print(f"  ❌ Failed: {result.comment}")

    def _close_trade(self, symbol, reason="AI_CLOSE"):
        positions = mt5.positions_get(magic=123456) or []
        for pos in positions:
            close_type = mt5.ORDER_TYPE_BUY if pos.type == 1 else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if close_type == mt5.ORDER_TYPE_BUY else tick.bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                "volume": pos.volume, "type": close_type,
                "position": pos.ticket, "price": price,
                "magic": 123456, "comment": reason,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(req)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  ✅ CLOSED {symbol}  reason={reason}  pnl={pos.profit:.2f}")
                self._open_ticket = None
            else:
                print(f"  ❌ Close failed: {result.comment}")

    def _log(self, symbol, direction, entry, sl, tp, sentiment, risk_mult):
        pd.DataFrame([{
            "Time": datetime.utcnow(), "Symbol": symbol,
            "Direction": direction, "Entry": entry,
            "SL": sl, "TP": tp, "Sentiment": sentiment,
            "RiskMult": risk_mult, "Daily": self.trades_today,
        }]).to_csv("trade_log.csv", mode='a', header=False, index=False)

    # ── H1 layer: sets directional bias ───────────────────────────────────────
    def _h1_scan(self, now_t, sess_name, sess):
        print(f"\n── H1 [{now_t.strftime('%H:%M')} UTC]  {sess_name} ──")
        sent = self._get_sentiment()
        sym  = self._best_pair()
        if sym is None: return

        result = self._build_state(sym, sent, sess["is_primary"], now_t)
        if result is None: return
        state, position, row, atr = result

        action = self.agent.act(state, position=position)
        label  = ["HOLD","BUY","SELL","CLOSE"][action]
        thresh = sess["sent_thresh"]
        print(f"  {sym}  AI={label}  Sent={sent:+.3f}")

        if action == 1 and sent > thresh:
            self._h1_bias = 1; self._h1_pair = sym
            print(f"  ✅ LONG BIAS → {sym}")
        elif action == 2 and sent < -thresh:
            self._h1_bias = -1; self._h1_pair = sym
            print(f"  ✅ SHORT BIAS → {sym}")
        else:
            self._h1_bias = 0; self._h1_pair = None

        self._h1_sess_mult = sess["risk_mult"]

        # Non-primary sessions: enter directly at H1 close
        if not sess["is_primary"] and self._h1_bias != 0:
            if self.trades_today < MAX_DAILY:
                direction = "BUY" if self._h1_bias == 1 else "SELL"
                self._open_trade(sym, direction, float(row['ATR']),
                                 self._h1_sess_mult, sent)
                self._h1_bias = 0

    # ── M15 layer: entry timing + exit management ─────────────────────────────
    def _m15_scan(self, now_t):
        sym = self._h1_pair
        result = self._build_state(
            sym or self.target_pairs[0], self._sentiment,
            True, now_t)
        if result is None: return
        state, position, row, atr = result

        # Use actual symbol from open position if one exists
        positions = mt5.positions_get(magic=123456) or []
        actual_sym = positions[0].symbol if positions else sym

        action = self.agent.act(state, position=position)
        label  = ["HOLD","BUY","SELL","CLOSE"][action]
        print(f"\n  ── M15 [{now_t.strftime('%H:%M')} UTC]  "
              f"pos={position}  AI={label} ──")

        # If in a trade and model says CLOSE
        if position != 0 and action == 3:
            print(f"  🧠 Model decided to CLOSE {actual_sym}")
            self._close_trade(actual_sym, reason="AI_CLOSE")

        # If flat and model agrees with H1 bias
        elif position == 0 and self._h1_bias != 0:
            if self.trades_today >= MAX_DAILY:
                print(f"  🛑 Daily cap"); return
            if action == 1 and self._h1_bias == 1:
                self._open_trade(actual_sym or sym, "BUY", float(row['ATR']),
                                 self._h1_sess_mult, self._sentiment)
            elif action == 2 and self._h1_bias == -1:
                self._open_trade(actual_sym or sym, "SELL", float(row['ATR']),
                                 self._h1_sess_mult, self._sentiment)

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print("🤖 Full Trader Bot running. Model decides entry AND exit.\n")
        while True:
            now   = datetime.utcnow()
            now_t = now.time()
            now_h = now.hour
            now_m = now.minute

            if now_t < dtime(0,2) and self.trades_today > 0:
                print("🔄 Midnight reset.")
                self.trades_today = 0; self._h1_bias = 0

            sess_name, sess = self._active_session(now_t)

            # H1 scan
            if now_m < 2 and now_h != self._last_h1:
                self._last_h1 = now_h
                if sess_name:
                    self._h1_scan(now_t, sess_name, sess)
                else:
                    # Dead zone — but still check if we need to close an open trade
                    positions = mt5.positions_get(magic=123456) or []
                    if positions:
                        result = self._build_state(
                            positions[0].symbol, self._sentiment, False, now_t)
                        if result:
                            state, position, row, atr = result
                            action = self.agent.act(state, position=position)
                            if action == 3:
                                print(f"  🧠 Dead zone — model chose to CLOSE")
                                self._close_trade(positions[0].symbol, "AI_DEADZONE")
                    self._h1_bias = 0

            # M15 scan (PRIMARY only)
            m15_slot = (now_m // 15) * 15
            if (sess_name == "PRIMARY" and now_m % 15 < 2
                    and m15_slot != self._last_m15 and now_m >= 2):
                self._last_m15 = m15_slot
                if self._h1_pair or (mt5.positions_get(magic=123456) or []):
                    self._m15_scan(now_t)

            time.sleep(30)


if __name__ == "__main__":
    bot = OverlapSniperBot()
    bot.run()