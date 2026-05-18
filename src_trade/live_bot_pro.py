import time
from datetime import datetime, time as dtime, timedelta

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
                  "risk_mult": 1.00, "sent_thresh": 0.10,
                  "is_primary": True,  "m15_scan": True},
    "SECONDARY": {"start": dtime( 7,0), "end": dtime(12,0),
                  "risk_mult": 0.50, "sent_thresh": 0.20,
                  "is_primary": False, "m15_scan": False},
    "TERTIARY":  {"start": dtime(17,0), "end": dtime(22,0),
                  "risk_mult": 0.25, "sent_thresh": 0.35,
                  "is_primary": False, "m15_scan": False},
}

SPIKE_THRESHOLD  = 2.5   
M15_FIB_WINDOW   = 0.5   
MIN_RR           = 1.5
MAX_DAILY_TRADES = 3


class OverlapSniperBot:

    def __init__(self):
        if not mt5.initialize():
            print("❌ MT5 init failed"); quit()

        self.client       = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.target_pairs = ["GBPUSD", "EURUSD", "USDJPY"]
        self.base_risk    = 0.01
        self.max_daily    = MAX_DAILY_TRADES

        self.trades_today        = 0
        self._last_sent_hour     = -1
        self._cached_sentiment   = 0.0

        # Layer 1 output — persists until next H1 candle
        self.h1_bias             = 0      #  1=long  -1=short  0=none
        self.h1_bias_pair        = None   # which pair the bias is on
        self.h1_bias_hour        = -1     # hour the bias was set
        self.h1_session_mult     = 1.0
        self.h1_sent_thresh      = 0.10

        # Layer 2 state
        self._last_m15_candle    = -1     # minute of last M15 scan (0,15,30,45)
        self._last_h1_candle     = -1     # hour of last H1 scan
        self._spike_cooldown     = 0      # M15 candles remaining in spike lockout
        self._position_open      = False  # simple flag; real check via MT5

        # Load model (46 features)
        self.agent = TradingDQNAgent(input_size=46)
        self.agent.load(config.MODEL_PATH)
        self.agent.epsilon = 0.0

        print("✅ Bot initialised.  Two-layer scanning active.")
        print(f"   H1  → directional bias  (all sessions)")
        print(f"   M15 → entry timing      (PRIMARY 13–17 UTC only)")
        print(f"   Spike protection: candles >{SPIKE_THRESHOLD}×ATR trigger 30-min lockout\n")

    def _active_session(self, now: dtime):
        for name, s in SESSIONS.items():
            if s["start"] <= now < s["end"]:
                return name, s
        return None, None

    def _get_sentiment(self) -> float:
        hour = datetime.utcnow().hour
        if hour == self._last_sent_hour:
            return self._cached_sentiment
        try:
            print("  🔍 Gemini sentiment refresh…")
            resp = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=(
                    "Analyse the last 4 hours of economic news for USD, EUR, GBP. "
                    "Reply with ONE float between -1.0 (Bearish) and 1.0 (Bullish). "
                    "No explanation — just the number."
                )
            )
            val = float(resp.text.strip())
            self._cached_sentiment = val
            self._last_sent_hour   = hour
            print(f"  📊 Sentiment: {val:+.3f}")
            return val
        except Exception as exc:
            print(f"  ⚠️  Gemini error: {exc}")
            return self._cached_sentiment

    def _best_pair(self) -> 'str | None':
        """Pick the pair with highest relative ATR on M15."""
        best, top = None, -1.0
        for sym in self.target_pairs:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 50)
            if rates is None: continue
            df = pd.DataFrame(rates)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(abs(df['high'] - df['close'].shift()),
                           abs(df['low']  - df['close'].shift())))
            rel = df['tr'].rolling(14).mean().iloc[-1] / \
                  (df['tr'].rolling(20).mean().iloc[-1] + 1e-8)
            if rel > top:
                top, best = rel, sym
        return best

    def _has_open_position(self) -> bool:
        positions = mt5.positions_get(magic=123456)
        return len(positions) > 0 if positions else False

    # ── Layer 1: H1 directional bias ──────────────────────────────────────────
    def _run_h1_layer(self, now_t: dtime, sess_name: str, sess: dict):
        """
        Runs once per new H1 candle.
        Reads the full H1 state, asks the AI for BUY/SELL/HOLD,
        applies sentiment gate, stores the result as self.h1_bias.
        """
        print(f"\n── H1 LAYER  [{now_t.strftime('%H:%M')} UTC]  {sess_name} ──")

        sentiment = self._get_sentiment()
        symbol    = self._best_pair()
        if symbol is None:
            print("  ⚠️  No valid pair found.")
            self.h1_bias = 0; return

        # Build H1 state
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 300)
        df    = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.columns = ['Open','High','Low','Close','TickVol','Spread','RealVol']

        tmp   = ForexEnv(df)
        state = tmp._get_state()
        state[44] = 1.0 if sess["is_primary"] else 0.0
        state[45] = float(sentiment)

        action = self.agent.act(state)
        label  = ["HOLD", "BUY", "SELL"][action]
        sent_thresh = sess["sent_thresh"]

        print(f"  Pair: {symbol}  AI: {label}  Sentiment: {sentiment:+.3f}  "
              f"Thresh: ±{sent_thresh}")

        # Sentiment gate
        if action == 1 and sentiment > sent_thresh:
            self.h1_bias      = 1
            self.h1_bias_pair = symbol
            print(f"  ✅ H1 LONG BIAS set on {symbol}")
        elif action == 2 and sentiment < -sent_thresh:
            self.h1_bias      = -1
            self.h1_bias_pair = symbol
            print(f"  ✅ H1 SHORT BIAS set on {symbol}")
        else:
            self.h1_bias      = 0
            self.h1_bias_pair = None
            print(f"  ➖ No bias  (AI={label}, sentiment={sentiment:+.3f})")

        self.h1_bias_hour    = datetime.utcnow().hour
        self.h1_session_mult = sess["risk_mult"]
        self.h1_sent_thresh  = sent_thresh

        # For SECONDARY / TERTIARY sessions: if bias is set, enter directly
        # on the H1 close — no M15 timing available in these sessions.
        if not sess["m15_scan"] and self.h1_bias != 0:
            self._attempt_entry(
                symbol    = self.h1_bias_pair,
                direction = "BUY" if self.h1_bias == 1 else "SELL",
                risk_mult = self.h1_session_mult,
                sentiment = sentiment,
                source    = "H1-direct"
            )

    # ── Layer 2: M15 entry timing ──────────────────────────────────────────────
    def _run_m15_layer(self, now_t: dtime):
        """
        Runs once per new M15 candle during the PRIMARY session.
        Only acts if H1 has set a bias this hour.
        Checks: spike protection → pattern confirmation → Fib proximity → entry.
        """
        # No bias from H1 this hour → nothing to do
        if self.h1_bias == 0 or self.h1_bias_pair is None:
            return

        # Bias must be from the current hour (don't carry yesterday's bias)
        if self.h1_bias_hour != datetime.utcnow().hour:
            self.h1_bias = 0; return

        if self._has_open_position():
            return   # already in a trade

        if self.trades_today >= self.max_daily:
            return

        symbol = self.h1_bias_pair
        print(f"\n  ── M15 LAYER  [{now_t.strftime('%H:%M')} UTC]  "
              f"{'LONG' if self.h1_bias==1 else 'SHORT'} on {symbol} ──")

        # ── Spike protection ───────────────────────────────────────────────
        if self._spike_cooldown > 0:
            self._spike_cooldown -= 1
            print(f"  🚫 Spike cooldown active — {self._spike_cooldown} M15 candles remaining.")
            return

        m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 30)
        if m15 is None: return
        m15_df       = pd.DataFrame(m15)
        atr_m15      = float((m15_df['high'] - m15_df['low']).rolling(14).mean().iloc[-1])
        last_range   = float(m15_df['high'].iloc[-1] - m15_df['low'].iloc[-1])
        spike_ratio  = last_range / (atr_m15 + 1e-8)

        if spike_ratio > SPIKE_THRESHOLD:
            self._spike_cooldown = 2   # wait 2 M15 candles (30 min) before entering
            print(f"  ⚡ SPIKE DETECTED  candle={last_range:.5f}  "
                  f"ATR={atr_m15:.5f}  ratio={spike_ratio:.1f}x  "
                  f"→ 30-min lockout activated.")
            return

        # ── Fib 61.8% proximity check ──────────────────────────────────────
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
        h1_df    = pd.DataFrame(h1_rates)
        sh50     = float(h1_df['high'].rolling(50).max().iloc[-1])
        sl50     = float(h1_df['low'].rolling(50).min().iloc[-1])
        atr_h1   = float((h1_df['high'] - h1_df['low']).rolling(14).mean().iloc[-1])
        fib_range = sh50 - sl50

        current_price = float(mt5.symbol_info_tick(symbol).bid)

        if self.h1_bias == 1:   # LONG: look for pullback to 61.8% (buy zone)
            fib618     = sh50 - fib_range * 0.618
            fib_dist   = abs(current_price - fib618)
            near_fib   = fib_dist < atr_h1 * M15_FIB_WINDOW
        else:                   # SHORT: look for rally to 61.8% (sell zone)
            fib618     = sl50 + fib_range * 0.618
            fib_dist   = abs(current_price - fib618)
            near_fib   = fib_dist < atr_h1 * M15_FIB_WINDOW

        print(f"  Fib 61.8%={fib618:.5f}  Price={current_price:.5f}  "
              f"Dist={fib_dist:.5f}  NearFib={'YES' if near_fib else 'NO'}")

        # ── M15 candlestick pattern confirmation ───────────────────────────
        last        = m15_df.iloc[-1]
        prev        = m15_df.iloc[-2]
        body        = abs(last['close'] - last['open'])
        lower_wick  = min(last['open'], last['close']) - last['low']
        upper_wick  = last['high'] - max(last['open'], last['close'])
        body_atr    = body        / (atr_m15 + 1e-8)
        lw_atr      = lower_wick  / (atr_m15 + 1e-8)
        uw_atr      = upper_wick  / (atr_m15 + 1e-8)
        bull_candle = last['close'] > last['open']
        bear_candle = last['close'] < last['open']

        hammer       = lw_atr > 2 * body_atr and uw_atr < 0.5 * body_atr and body_atr > 0
        bull_engulf  = (bull_candle
                        and not (prev['close'] > prev['open'])
                        and last['close'] > prev['open']
                        and last['open']  < prev['close'])
        shoot_star   = uw_atr > 2 * body_atr and lw_atr < 0.5 * body_atr and body_atr > 0
        bear_engulf  = (bear_candle
                        and (prev['close'] > prev['open'])
                        and last['close'] < prev['open']
                        and last['open']  > prev['close'])

        if self.h1_bias == 1:
            pattern_ok = hammer or bull_engulf
            pattern_name = "Hammer" if hammer else ("Bull Engulf" if bull_engulf else "None")
        else:
            pattern_ok = shoot_star or bear_engulf
            pattern_name = "Shoot Star" if shoot_star else ("Bear Engulf" if bear_engulf else "None")

        print(f"  M15 Pattern: {pattern_name}  Confirmed: {'YES' if pattern_ok else 'NO'}")


        if near_fib and pattern_ok:
            print(f"  🎯 PERFECT SETUP: Fib 61.8% + Pattern confirmation")
            self._attempt_entry(
                symbol    = symbol,
                direction = "BUY" if self.h1_bias == 1 else "SELL",
                risk_mult = self.h1_session_mult,
                sentiment = self._cached_sentiment,
                source    = "M15-fib+pattern"
            )
        elif near_fib and not pattern_ok:
            print(f"  📍 Near Fib but no pattern — waiting for confirmation.")
        elif pattern_ok and not near_fib:
            print(f"  📍 Pattern confirmed but not at Fib level — "
                  f"price={current_price:.5f}  Fib={fib618:.5f}  "
                  f"(dist={fib_dist:.5f} > threshold {atr_h1*M15_FIB_WINDOW:.5f})")
        else:
            print(f"  ➖ No setup — waiting.")

    # ── shared entry logic ────────────────────────────────────────────────────
    def _attempt_entry(self, symbol: str, direction: str,
                       risk_mult: float, sentiment: float, source: str):
        """
        Final checks and order placement.
        Computes structure SL/TP, verifies RR, sizes position, sends order.
        """
        if self._has_open_position():
            print(f"  ⚠️  Position already open — skipping."); return
        if self.trades_today >= self.max_daily:
            print(f"  🛑 Daily cap reached."); return

        tick  = mt5.symbol_info_tick(symbol)
        entry = tick.ask if direction == "BUY" else tick.bid

        # Structure SL/TP from H1 swing highs/lows
        sl, tp, rr = self._compute_structure_sl_tp(symbol, direction, entry)

        if rr < MIN_RR:
            print(f"  ⚠️  RR={rr:.2f} < {MIN_RR} — bad structure, skip."); return

        lots = self._compute_lot_size(symbol, entry, sl, risk_mult)

        print(f"\n  {'🚀' if direction=='BUY' else '📉'} ENTRY [{source}]")
        print(f"     {direction} {symbol} @ {entry:.5f}")
        print(f"     SL={sl:.5f}  TP={tp:.5f}  RR={rr:.2f}  Lots={lots}")
        print(f"     Sentiment={sentiment:+.3f}  Session risk={risk_mult*100:.0f}%")

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lots,
            "type":         mt5.ORDER_TYPE_BUY if direction=="BUY"
                            else mt5.ORDER_TYPE_SELL,
            "price":        entry,
            "sl":           round(sl, 5),
            "tp":           round(tp, 5),
            "magic":        123456,
            "comment":      f"Sniper|{source}|RR{rr:.1f}|S{sentiment:.2f}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ Order placed successfully  (ticket #{result.order})")
            self.trades_today += 1
            self._log(symbol, direction, entry, sl, tp, rr, sentiment,
                      risk_mult, source)
        else:
            print(f"  ❌ Order failed ({result.retcode}): {result.comment}")

    # ── SL/TP from market structure ───────────────────────────────────────────
    def _compute_structure_sl_tp(self, symbol: str, direction: str,
                                  entry: float) -> 'tuple[float,float,float]':
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 120)
        df    = pd.DataFrame(rates)
        atr   = float((df['high'] - df['low']).rolling(14).mean().iloc[-1])
        sh50  = float(df['high'].rolling(50).max().iloc[-1])
        sl50  = float(df['low'].rolling(50).min().iloc[-1])
        sh100 = float(df['high'].rolling(100).max().iloc[-1])
        sl100 = float(df['low'].rolling(100).min().iloc[-1])

        if direction == "BUY":
            sl_p    = sl50  - atr * 0.5
            tp_p    = sh50  - atr * 0.2
            sl_dist = entry - sl_p
            tp_dist = tp_p  - entry
            if sl_dist <= 0 or tp_dist / (sl_dist + 1e-8) < MIN_RR:
                tp_p    = sh100 - atr * 0.2
                tp_dist = tp_p  - entry
        else:
            sl_p    = sh50  + atr * 0.5
            tp_p    = sl50  + atr * 0.2
            sl_dist = sl_p  - entry
            tp_dist = entry - tp_p
            if sl_dist <= 0 or tp_dist / (sl_dist + 1e-8) < MIN_RR:
                tp_p    = sl100 + atr * 0.2
                tp_dist = entry - tp_p

        rr = tp_dist / (sl_dist + 1e-8)
        return sl_p, tp_p, max(rr, 0.0)

    def _compute_lot_size(self, symbol: str, entry: float,
                           sl: float, risk_mult: float) -> float:
        acct     = mt5.account_info()
        risk_amt = acct.balance * self.base_risk * risk_mult
        sl_dist  = abs(entry - sl)
        is_jpy   = 'JPY' in symbol
        pip_size = 0.01   if is_jpy else 0.0001
        pip_val  = 0.091  if is_jpy else 10.0
        sl_pips  = sl_dist / pip_size
        lot      = risk_amt / (sl_pips * pip_val + 1e-8)
        return round(max(0.01, min(lot, 5.0)), 2)

    def _log(self, symbol, direction, entry, sl, tp, rr,
             sentiment, risk_mult, source):
        pd.DataFrame([{
            "Timestamp":  datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol":     symbol,
            "Direction":  direction,
            "Source":     source,
            "Entry":      round(entry, 5),
            "SL":         round(sl, 5),
            "TP":         round(tp, 5),
            "RR":         round(rr, 2),
            "Sentiment":  sentiment,
            "RiskMult":   risk_mult,
            "DailyCount": self.trades_today,
        }]).to_csv("sniper_trade_log.csv", mode='a', header=False, index=False)

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print("🤖 Sniper Bot running.  Two-layer scanning active.\n")

        while True:
            now   = datetime.utcnow()
            now_t = now.time()
            now_h = now.hour
            now_m = now.minute

            # Midnight reset
            if now_t < dtime(0, 2) and self.trades_today > 0:
                print("\n🔄 Midnight UTC — resetting daily counters.")
                self.trades_today = 0
                self.h1_bias      = 0

            sess_name, sess = self._active_session(now_t)

            # ── LAYER 1: H1 scan (once per new H1 candle, any session) ───────
            if now_m < 2 and now_h != self._last_h1_candle:
                self._last_h1_candle = now_h
                if sess_name is not None:
                    self._run_h1_layer(now_t, sess_name, sess)
                else:
                    print(f"\n💤 [{now_t.strftime('%H:%M')} UTC] Dead zone — no scan.")
                    self.h1_bias = 0

            # ── LAYER 2: M15 scan (once per new M15 candle, PRIMARY only) ────
            m15_slot = (now_m // 15) * 15   # 0, 15, 30, or 45
            if (sess_name == "PRIMARY"
                    and now_m % 15 < 2          # first 2 minutes of each M15 candle
                    and m15_slot != self._last_m15_candle
                    and now_m >= 2):             # don't double-fire with H1 at :00
                self._last_m15_candle = m15_slot
                self._run_m15_layer(now_t)

            time.sleep(30)


if __name__ == "__main__":
    bot = OverlapSniperBot()
    bot.run()