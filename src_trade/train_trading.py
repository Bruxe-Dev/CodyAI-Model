import numpy as np
import os
from src_trade.trading_env import ForexEnv, download_forex_data
from src_trade.trading_agent import TradingDQNAgent

# ANSI Colors for Terminal
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def train_forex_agent(pair='EURUSD=X', episodes=2000):
    # 1. FORCE FRESH START
    if os.path.exists(f"best_{pair}.npz"):
        os.remove(f"best_{pair}.npz")
        print("🗑️ Deleted old model to force fresh learning.")

    data = download_forex_data(pair='EURUSD', start_year=2019, end_year=2022)
    env = ForexEnv(data)
    agent = TradingDQNAgent(input_size=31)
    
    # 2. UNLEASH EXPLORATION
    agent.epsilon = 1.0 
    agent.epsilon_decay = 0.999 # Stays curious longer
    best_return = -float('inf')

    print(f"\n{'Ep':<5} | {'Return %':<10} | {'Portfolio':<12} | {'Trades':<8} | {'Eps'}")
    print("-" * 60)

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        trades_today = 0
        
        while not done:
            is_overlap = state[29] # FIX: Correct index for Overlap
            
            # Allow up to 3 trades per session to maximize profit potential
            if is_overlap == 1.0 and trades_today < 3 and env.position == 0:
                action = agent.act(state)
            elif env.position != 0:
                action = agent.act(state) # AI decides exit
            else:
                action = 0 
                
            next_state, reward, done = env.step(action)
            
            # Reward "Greed" for high-profit trades
            if reward > 0: reward *= 2.0 

            if action != 0 and env.position != 0:
                trades_today += 1

            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
        
        ep_return = env.get_return()
        port_val = env.get_portfolio_value()
        
        # COLOR OUTPUT: Green for Profit, Red for Loss
        color = GREEN if ep_return > 0 else (RED if ep_return < 0 else RESET)
        
        if ep_return > best_return:
            best_return = ep_return
            agent.save(f"best_{pair}.npz")
        
        if ep % 10 == 0 or ep == 1:
            print(f"{ep:<5} | {color}{ep_return:>8.2f}%{RESET} | ${port_val:>10.2f} | {env.total_trades:<8} | {agent.epsilon:.4f}")

if __name__ == "__main__":
    train_forex_agent()